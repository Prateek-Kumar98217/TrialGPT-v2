import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from dataclasses import asdict
from .model import GPTConfig, TrialGPT

if __name__ == "__main__":
    # --- Configuration ---
    gradient_accumulation_steps = 32
    learning_rate = 6e-5
    dataset = 'openwebtext'
    init_from = 'resume'  # 'scratch' or 'resume'
    max_iterations = 50000
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    evaluation_interval = 100
    loss_sampling = 1000
    batch_size = 18
    dropout = 0.1
    dtype = 'float16'
    out_dir = 'output'
    eval_only = False
    always_save_checkpoint = False
    grad_clip = 1.0

    config = GPTConfig(
        dim=1024,
        num_heads=8,
        num_layers=12,
        num_kv_heads=2,
        max_seq_length=512,
        vocab_size=50257,
        norm_type='rms'
    )

    data_dir = os.path.join('data', dataset)

    def get_batch(split):
        data = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - config.max_seq_length, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i + config.max_seq_length]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + config.max_seq_length]).astype(np.int64)) for i in ix])
        return x.to(device), y.to(device)

    iter_num = 0
    best_val_loss = 1e9

    if init_from == 'scratch':
        print("Initializing TrialGPT from scratch")
        model = TrialGPT(config)
        if torch.__version__ >= '2.0.0' and device == 'cuda':
          print("Compiling model for graph-based optimization...")
          model = torch.compile(model)  # Use mode='reduce-overhead' for small models
  
    elif init_from == 'resume':
        print(f"Resuming from checkpoint in {out_dir}")
        ckpt = torch.load(os.path.join(out_dir, 'ckpt.pt'), map_location=device)
        config = GPTConfig(**ckpt['config'])
        model = TrialGPT(config)
        model.load_state_dict(ckpt['model'])
        iter_num = ckpt['iter_num']
        best_val_loss = ckpt['best_val_loss']

    model.to(device)
    scaler = torch.GradScaler("cuda", enabled=(dtype == 'float16'))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)

    if init_from == 'resume':
        optimizer.load_state_dict(ckpt['optimizer'])

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(loss_sampling)
            for k in tqdm(range(loss_sampling), desc=f"Evaluating {split}"):
                X, Y = get_batch(split)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(X)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=-1)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    scheduler = OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=max_iterations,
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=30.0,
        final_div_factor=1e4,
    )

    while iter_num < max_iterations:
        if iter_num % evaluation_interval == 0 or eval_only:
            print("Evaluating...")
            losses = estimate_loss()
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num >= 0:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': asdict(config),
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': asdict(config),
                    }
                    print(f"Saving checkpoint to {out_dir}")
                    os.makedirs(out_dir, exist_ok=True)
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

        if eval_only:
            print("Running in evaluation-only mode...")
            losses = estimate_loss()
            print(f"Eval-only: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            exit()
        running_loss=0
        pbar=tqdm(range(gradient_accumulation_steps), desc=f"Step {iter_num}")
        for micro_step in pbar:
            X, Y = get_batch('train')
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(X)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=-1)
                loss = loss / gradient_accumulation_steps
            scaler.scale(loss).backward()

            if micro_step + 1 == gradient_accumulation_steps:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
        iter_num += 1

        if iter_num >= max_iterations:
            break
