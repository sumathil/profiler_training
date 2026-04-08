#!/usr/bin/env python3
import argparse
import csv
import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, seq_len, d_model, nhead, num_layers, ff_dim, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids):
        bsz, seqlen = token_ids.shape
        pos = torch.arange(seqlen, device=token_ids.device).unsqueeze(0).expand(bsz, seqlen)
        x = self.token_emb(token_ids) + self.pos_emb(pos)
        x = self.encoder(x)
        x = self.norm(x)
        return self.head(x)


def parse_args():
    parser = argparse.ArgumentParser(description="Synthetic Transformer LM training benchmark with AMP")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--ff-dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--use-profiler", action="store_true", help="Enable torch.profiler")
    parser.add_argument("--profiler-dir", default="./profiles", help="Profiler output directory")
    parser.add_argument("--profiler-chrome-trace", action="store_true", help="Write chrome trace JSON")
    parser.add_argument("--profiler-chrome-trace-file", default="trace.json")
    parser.add_argument("--profiler-wait", type=int, default=1)
    parser.add_argument("--profiler-warmup", type=int, default=1)
    parser.add_argument("--profiler-active", type=int, default=3)
    parser.add_argument("--profiler-repeat", type=int, default=1)
    parser.add_argument("--results-json", default="", help="Optional path to write run metrics as JSON")
    parser.add_argument("--results-csv", default="", help="Optional path to append run metrics as CSV")
    return parser.parse_args()


def resolve_device(arg):
    if arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg


def write_results(args, metrics):
    if args.results_json:
        with open(args.results_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
            f.write("\n")

    if args.results_csv:
        file_exists = os.path.exists(args.results_csv)
        with open(args.results_csv, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            if (not file_exists) or os.path.getsize(args.results_csv) == 0:
                writer.writeheader()
            writer.writerow(metrics)


def main():
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be > 0")
    if args.seq_len <= 0 or args.batch_size <= 0:
        raise ValueError("--seq-len and --batch-size must be > 0")
    if args.vocab_size <= 1:
        raise ValueError("--vocab-size must be > 1")
    if args.d_model <= 0 or args.num_layers <= 0 or args.ff_dim <= 0:
        raise ValueError("--d-model, --num-layers, and --ff-dim must be > 0")
    if args.d_model % args.nhead != 0:
        raise ValueError("--d-model must be divisible by --nhead")

    torch.manual_seed(args.seed)
    device = torch.device(resolve_device(args.device))
    use_amp = device.type == "cuda" and not args.disable_amp

    profiler = None
    if args.use_profiler:
        os.makedirs(args.profiler_dir, exist_ok=True)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        trace_handler = torch.profiler.tensorboard_trace_handler(args.profiler_dir)
        if args.profiler_chrome_trace:
            def trace_handler(prof):
                base, ext = os.path.splitext(args.profiler_chrome_trace_file)
                ext = ext if ext else ".json"
                trace_path = os.path.join(args.profiler_dir, f"{base}{ext}")
                if os.path.exists(trace_path) or args.profiler_repeat > 1:
                    trace_path = os.path.join(args.profiler_dir, f"{base}_step{prof.step_num}{ext}")
                prof.export_chrome_trace(trace_path)
        profiler = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=args.profiler_wait,
                warmup=args.profiler_warmup,
                active=args.profiler_active,
                repeat=args.profiler_repeat,
            ),
            on_trace_ready=trace_handler,
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        )
        profiler.start()

    model = TransformerLM(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    model.train()
    start_time = time.time()
    tokens_per_step = args.batch_size * args.seq_len

    for step in range(1, args.steps + 1):
        token_ids = torch.randint(
            low=0,
            high=args.vocab_size,
            size=(args.batch_size, args.seq_len),
            device=device,
        )
        targets = torch.randint(
            low=0,
            high=args.vocab_size,
            size=(args.batch_size, args.seq_len),
            device=device,
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(token_ids)
            loss = criterion(logits.reshape(-1, args.vocab_size), targets.reshape(-1))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if profiler is not None:
            profiler.step()
        if step == args.warmup_steps and device.type == "cuda":
            torch.cuda.synchronize()

    if device.type == "cuda":
        torch.cuda.synchronize()
    if profiler is not None:
        profiler.stop()

    elapsed = time.time() - start_time
    step_s = elapsed / args.steps
    toks_s = (tokens_per_step * args.steps) / elapsed
    metrics = {
        "script": "transformer_lm_amp",
        "device": str(device),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "tokens_per_step": tokens_per_step,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "ff_dim": args.ff_dim,
        "amp_enabled": int(bool(use_amp)),
        "elapsed_s": round(elapsed, 6),
        "step_s": round(step_s, 6),
        "tokens_s": round(toks_s, 2),
    }
    print(
        f"steps={args.steps} elapsed_s={elapsed:.4f} step_s={step_s:.6f} "
        f"tokens_per_step={tokens_per_step} tokens_s={toks_s:.2f}"
    )
    write_results(args, metrics)


if __name__ == "__main__":
    main()
