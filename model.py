import torch
import torch.nn as nn
import math
from dataclasses import dataclass

@dataclass
class GPTConfig():
    dim: int
    num_heads: int = 8
    num_layers: int = 12
    norm_type: str = "rms"
    num_kv_heads: int = 2
    dropout:float =0.2
    max_seq_length: int = 1024
    vocab_size:int = 50257

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps= eps
        self.scale=nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        norm=x.norm(2, dim=-1, keepdim=True)
        return x/(norm+self.eps)*self.scale
    
class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()

        inv_freq=1.0/(10000 **(torch.arange(0, dim, 2).float()/dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len):
        pos=torch.arange(seq_len, dtype=torch.float, device=self.inv_freq.device)
        freqs=torch.einsum("i,j->ij", pos, self.inv_freq)
        return freqs

    def apply_rotary(self, x, rotray_emb):
        x1, x2=x[..., ::2],x[..., 1::2]
        cos, sin=rotray_emb.cos(), rotray_emb.sin()
        return torch.cat([x1*cos-x2*sin, x1*sin-x2*cos], dim=-1)
    
class MultiQueryAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads):
        super().__init__()

        self.num_heads=num_heads
        self.num_kv_heads=num_kv_heads
        self.head_dim=dim//num_heads

        self.q_proj=nn.Linear(dim, dim)
        self.k_proj=nn.Linear(dim, self.head_dim*self.num_kv_heads)
        self.v_proj=nn.Linear(dim, self.head_dim*self.num_kv_heads)
        self.output_proj=nn.Linear(dim, dim)

        self.rotary=RotaryEmbedding(self.head_dim)
    
    def forward(self, x):
        B,T,C=x.size()
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        rotary_emb=self.rotary(T).to(x.device)

        q=self.rotary.apply_rotary(q, rotary_emb.unsqueeze(0).unsqueeze(0))
        k=self.rotary.apply_rotary(k, rotary_emb.unsqueeze(0).unsqueeze(0))

        if self.num_heads != self.num_kv_heads:
            assert self.num_heads % self.num_kv_heads == 0
            repeat_factor = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        context = attn @ v

        out = context.transpose(1, 2).reshape(B, T, C)
        return self.output_proj(out)
    
class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.proj=nn.Linear(dim, dim*2)

    def forward(self, x):
        x_proj, gate = self.proj(x).chunk(2, dim=-1)
        return x_proj * torch.nn.functional.silu(gate)

class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.net=nn.Sequential(
            SwiGLU(dim),
            nn.Linear(dim, dim)
        )
    
    def forward(self, x):
        return self.net(x)
    
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, dropout=0.2, norm_type="rms", num_layers=1):
        super().__init__()

        self.scale=(2*num_layers)**0.5
        Norm = RMSNorm if norm_type=="rms" else nn.LayerNorm
        self.norm1=Norm(dim)
        self.attn=MultiQueryAttention(dim, num_heads, num_kv_heads)
        self.drop1=nn.Dropout(dropout)
        self.norm2=Norm(dim)
        self.ffn=FeedForward(dim)
        self.drop2=nn.Dropout(dropout)

    def forward(self, x):
        x= x+self.drop1(self.attn(self.norm1(x)))/self.scale
        x=x+ self.drop2(self.ffn(self.norm2(x)))/self.scale
        return x
    
class TrialGPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config=config
        self.embed=nn.Embedding(config.vocab_size, config.dim)
        self.blocks=nn.ModuleList([
            TransformerBlock(
                dim=config.dim,
                num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
                norm_type=config.norm_type,
                dropout=config.dropout,
                num_layers=config.num_layers
            ) for _ in range(config.num_layers)
        ])

        self.norm_final=RMSNorm(config.dim) if config.norm_type=="rms" else nn.LayerNorm(config.dim)
        self.lm_head=nn.Linear(config.dim, config.vocab_size, bias=False)
        self.lm_head.weight=self.embed.weight

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.config.max_seq_length, f"Sequence length {T} exceeds max {self.config.max_seq_length}"
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x)
        x = self.norm_final(x)
        logits = self.lm_head(x)
        return logits
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) < self.config.max_seq_length else idx[:, -self.config.max_seq_length:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                values, _ = torch.topk(logits, top_k)
                logits[logits < values[:, [-1]]] = -float('inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

        return idx