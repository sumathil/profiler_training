import argparse
import time

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="CUDA profiling example: batched matmul")
    parser.add_argument("--m", type=int, default=4096, help="Rows of A and C")
    parser.add_argument("--n", type=int, default=4096, help="Columns of B and C")
    parser.add_argument("--k", type=int, default=4096, help="Columns of A / rows of B")
    parser.add_argument("--batch", type=int, default=1, help="Batch size for batched matmul")
    parser.add_argument("--iters", type=int, default=200, help="Total iterations")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations (not profiled)")
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Data type for inputs",
    )
    parser.add_argument("--use-tf32", action="store_true", help="Enable TF32 for fp32 matmul")
    return parser.parse_args()


def make_dtype(dtype_str):
    if dtype_str == "fp16":
        return torch.float16
    if dtype_str == "bf16":
        return torch.bfloat16
    return torch.float32


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        return

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = make_dtype(args.dtype)

    if args.use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Allocate inputs
    a = torch.randn(args.batch, args.m, args.k, device=device, dtype=dtype)
    b = torch.randn(args.batch, args.k, args.n, device=device, dtype=dtype)

    # Warmup
    for _ in range(args.warmup):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()

    # Profile window
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push("matmul_profile_window")

    start = time.time()
    for i in range(args.iters):
        torch.cuda.nvtx.range_push(f"iter_{i}")
        _ = torch.matmul(a, b)
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    end = time.time()

    torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()

    elapsed_ms = (end - start) * 1000.0
    avg_ms = elapsed_ms / args.iters
    print(f"Elapsed: {elapsed_ms:.2f} ms, Avg per iter: {avg_ms:.3f} ms")


if __name__ == "__main__":
    main()

"""
Run with Nsight Systems:
    nsys profile -o output_profile python cuda_profile_matmul.py --m 4096 --n 4096 --k 4096 --iters 200

Notes:
- Use --dtype fp32 with --use-tf32 to observe TF32 behavior on Ampere+ GPUs.
- Increase --batch for larger GEMM workloads.
"""
