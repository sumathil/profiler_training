# Compiler-Guided Optimization with Nsight (Tutorial)

This tutorial walks through a repeatable workflow for **compiler-guided optimization** using
NVIDIA Nsight tools on the CUDA vector-add benchmark in this repo.

The goal is to:
- establish a baseline,
- use **Nsight Systems** to confirm where time is spent,
- use **Nsight Compute** to see kernel-level bottlenecks,
- feed **compiler feedback** (register count, spills, line mapping) into targeted changes,
- validate the impact with both the compiler output and performance measurements.

> Assumptions: You have the CUDA Toolkit, Nsight Systems, and Nsight Compute installed.

---

## 1) Build a baseline

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Run the baseline benchmark and save the results:

```bash
./build/benchmark | tee baseline.csv
```

Scripted version:

```bash
./scripts/run_baseline.sh baseline.csv 33554432 200
```

---

## 2) Add compiler feedback and line info

We want **line-level correlation** and **compiler feedback** (registers, spills) for later steps.
Create a separate build directory so baseline remains unchanged:

```bash
cmake -S . -B build-lineinfo \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_FLAGS="-lineinfo -Xptxas -v"
cmake --build build-lineinfo -j
```

Notes:
- `-lineinfo` lets Nsight map SASS/PTX back to source lines.
- `-Xptxas -v` prints **register usage** and **spill** info at compile time.

Capture the compiler feedback by rebuilding with `--verbose`:

```bash
cmake --build build-lineinfo -j --verbose | tee build-lineinfo.log
```

Scripted version:

```bash
./scripts/build_lineinfo.sh
```

Open `build-lineinfo.log` and find the `ptxas` output. You should see lines like:

```
ptxas info    : Used XX registers, YY bytes smem, ZZ bytes cmem[0], ...
ptxas info    : Function properties for vectorAdd
ptxas info    : ...
```

Write down:
- register count
- spill (if any)
- shared memory usage

These are the **compiler signals** that will guide optimization.

---

## 3) Profile system-level timing (Nsight Systems)

Run a short profile to verify GPU utilization and kernel launch behavior:

```bash
nsys profile -o nsys_baseline ./build-lineinfo/benchmark 33554432 200
```

Scripted version:

```bash
./scripts/profile_nsys.sh nsys_baseline 33554432 200
```

### Matmul comparison (naive vs tiled)

```bash
./scripts/profile_nsys_matmul.sh nsys_matmul 1024 50
./scripts/profile_nsys_matmul_tiled.sh nsys_matmul_tiled 1024 50
```

Then open the report in Nsight Systems and answer:
- Is the runtime dominated by the kernel, or by copies/launch overhead?
- Are there unexpected stalls or gaps between kernel launches?

If kernel time dominates, proceed to kernel analysis in Nsight Compute.
If launch overhead dominates, focus on batching or increasing work per launch.

---

## 4) Profile kernel behavior (Nsight Compute)

Collect a focused report for the kernel:

```bash
ncu --set full -o ncu_baseline ./build-lineinfo/benchmark 33554432 200
```

Scripted version:

```bash
./scripts/profile_ncu.sh ncu_baseline 33554432 200
```

### Matmul comparison (naive vs tiled)

```bash
./scripts/profile_ncu_matmul.sh ncu_matmul 1024 50
./scripts/profile_ncu_matmul_tiled.sh ncu_matmul_tiled 1024 50
```

Open the report in Nsight Compute and inspect:
- **Achieved Occupancy** and **SM utilization**
- **Memory throughput** vs theoretical bandwidth
- **Instruction mix** (FMA vs memory ops)
- **Warp stall reasons** (memory dependency, execution dependency)

Correlate the hottest lines in the Source view with compiler feedback:
- High register usage can reduce occupancy.
- Spills indicate register pressure and slow local memory accesses.

---

## 5) Create compiler-guided hypotheses

Use **compiler feedback + Nsight metrics** to form hypotheses. Examples:

1. **High register count, low occupancy**
   - Try smaller block sizes.
   - Consider limiting registers via `-maxrregcount`.

2. **Memory-bound kernel** (high memory stall)
   - Try ensuring coalesced accesses (already contiguous here).
   - Validate that block size is tuned to hardware.

3. **Spills detected**
   - Look for unnecessary temporaries.
   - Simplify expressions or reduce live ranges.

---

## 6) Apply a small, targeted change (optional)

A minimal code change that often reduces register pressure is adding `__restrict__`
to pointers to help the compiler with alias analysis.

Edit `benchmark.cu` and change the kernel signature to:

```cpp
__global__ void vectorAdd(const float *__restrict__ a,
                          const float *__restrict__ b,
                          float *__restrict__ c,
                          int n)
```

Then rebuild with line info and compare:

```bash
cmake --build build-lineinfo -j --verbose | tee build-lineinfo_restrict.log
ncu --set full -o ncu_restrict ./build-lineinfo/benchmark 33554432 200
```

Check if:
- register count changed
- spill count changed
- achieved occupancy and throughput improved

---

## 7) Validate against baseline

Run the benchmark again and compare output:

```bash
./build-lineinfo/benchmark | tee restrict.csv
```

Compare `baseline.csv` and `restrict.csv`:
- Did `AvgKernelMs` decrease?
- Did `EstimatedGBps` increase?

Use Nsight results to explain **why** the change helped or didn’t.

### Matmul benchmark scripts

```bash
./scripts/run_matmul.sh matmul.csv 1024 50
./scripts/run_matmul_tiled.sh matmul_tiled.csv 1024 50
```

---

## 8) Optional: Explore compiler constraints

To see how register limits affect occupancy:

```bash
cmake -S . -B build-rreg \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_FLAGS="-lineinfo -Xptxas -v -maxrregcount=64"
cmake --build build-rreg -j
ncu --set full -o ncu_rreg ./build-rreg/benchmark 33554432 200
```

Scripted version:

```bash
./scripts/build_rreg.sh 64
./scripts/profile_ncu.sh ncu_rreg 33554432 200
```

Tradeoff:
- Lower register count may increase occupancy,
- but can also cause spills. The Nsight Compute report will confirm.

---

## 9) Document the decision

For each change, record:
- the compiler feedback (registers, spills)
- the Nsight Compute metrics
- the benchmark timing change

This is the heart of **compiler-guided optimization**: the compiler tells you
where it struggled, and Nsight shows how that surfaced in hardware behavior.

---

## 10) Clean up

If you created multiple build folders:

```bash
rm -rf build-lineinfo build-rreg
```

---

## Quick Checklist

- [ ] Baseline timing recorded
- [ ] Compiler feedback captured (`-Xptxas -v`)
- [ ] Nsight Systems timeline inspected
- [ ] Nsight Compute metrics reviewed
- [ ] Hypothesis formed
- [ ] Small change applied
- [ ] Results validated and explained
