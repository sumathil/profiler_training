#!/usr/bin/env python3
import argparse
import csv
import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR10 ResNet18 training with AMP")
    parser.add_argument("--data-dir", default="./data", help="Dataset directory")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--max-steps", type=int, default=200, help="0 = full epoch")
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--channels-last", action="store_true")
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
    torch.manual_seed(args.seed)
    device = torch.device(resolve_device(args.device))
    use_profiler = args.use_profiler
    profiler = None

    if use_profiler:
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
                    trace_path = os.path.join(
                        args.profiler_dir,
                        f"{base}_step{prof.step_num}{ext}",
                    )
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

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_ds = datasets.CIFAR10(
        args.data_dir,
        train=True,
        download=not args.no_download,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    model = models.resnet18(num_classes=10)
    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    use_amp = (device.type == "cuda") and (not args.disable_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    model.train()
    step = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        for images, labels in train_loader:
            if args.channels_last and device.type == "cuda":
                images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            else:
                images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            step += 1
            if profiler is not None:
                profiler.step()
            if step == args.warmup_steps and device.type == "cuda":
                torch.cuda.synchronize()
            if args.max_steps and step >= args.max_steps:
                break
        scheduler.step()
        if args.max_steps and step >= args.max_steps:
            break

    if device.type == "cuda":
        torch.cuda.synchronize()
    if profiler is not None:
        profiler.stop()

    elapsed = time.time() - start_time
    steps_done = step
    if steps_done > 0:
        metrics = {
            "script": "cifar10_resnet_amp",
            "device": str(device),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "max_steps": args.max_steps,
            "channels_last": int(bool(args.channels_last)),
            "amp_enabled": int(bool(use_amp)),
            "steps": steps_done,
            "elapsed_s": round(elapsed, 6),
            "step_s": round(elapsed / steps_done, 6),
        }
        print(f"steps={steps_done} elapsed_s={elapsed:.4f} step_s={elapsed/steps_done:.6f}")
        write_results(args, metrics)


if __name__ == "__main__":
    main()
