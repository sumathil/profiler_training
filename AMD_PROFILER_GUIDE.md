# AMD GPU Profiler Guide

Complete guide for profiling AMD GPU applications using ROCm profiling tools.

## Available Tools (from rocm-systems)

| Tool | Also Known As | Purpose | Installation | When to Use |
|------|---------------|---------|--------------|-------------|
| **rocprof** | rocprof | Basic profiling, stats, timeline | Included with ROCm | Quick checks, stats, always available |
| **rocprofiler-systems** | Omnitrace (old) | Timeline/system profiling | Separate install | Detailed timeline, Python, system-wide (like Nsight Systems) |
| **rocprofiler-compute** | Omniperf (old) | Kernel analysis | Separate install | Deep kernel metrics, roofline, optimization (like Nsight Compute) |

**Note**: rocprofiler-systems and rocprofiler-compute are both from the [rocm-systems repository](https://github.com/ROCm/rocm-systems).

## Tool 1: rocprof (Always Available)

rocprof is included with ROCm - no installation needed.

### Basic Usage

```bash
# Quick kernel statistics
rocprof --stats ./app

# Timeline trace
rocprof --sys-trace --hip-trace ./app

# Collect specific metrics
rocprof -i metrics.txt ./app

# Profile Python/PyTorch
rocprof --hip-trace python train.py
```

### Creating a Metrics File

Create `metrics.txt` with hardware counters you want:

```bash
cat > metrics.txt << 'EOF'
# Compute metrics
pmc: SQ_WAVES
pmc: SQ_INSTS_VALU
pmc: SQ_INSTS_SALU

# Memory metrics
pmc: TCC_HIT
pmc: TCC_MISS
pmc: FETCH_SIZE
pmc: WRITE_SIZE
EOF

rocprof -i metrics.txt ./app
```

### List Available Metrics

```bash
# See all available metrics
rocprof --list-basic
rocprof --list-derived

# Search for specific metrics
rocprof --list-basic | grep -i cache
rocprof --list-basic | grep -i memory
```

### Output Files

rocprof generates:
- `results.stats.csv` - Kernel statistics
- `results.hip_stats.csv` - HIP API statistics  
- `results.json` - Timeline trace (view in Chrome Tracing or Perfetto)

### View Results

```bash
# View stats
cat results.stats.csv
cat results.hip_stats.csv

# Timeline trace
# Open chrome://tracing in browser and load results.json
# Or upload to https://ui.perfetto.dev
```

## Tool 2: rocprofiler-systems (Timeline Profiling)

rocprofiler-systems (formerly Omnitrace) provides advanced timeline profiling with better visualization - like NVIDIA Nsight Systems.

### Installation (No Sudo Required)

```bash
# Download installer
wget https://github.com/ROCm/rocprofiler-systems/releases/latest/download/rocprofiler-systems-install.py

# Install to ~/.local (no sudo needed)
python3 ./rocprofiler-systems-install.py --prefix ~/.local --rocm 6.2

# Setup environment
source ~/.local/share/rocprofiler-systems/setup-env.sh

# Add to ~/.bashrc for permanent setup
echo "source ~/.local/share/rocprofiler-systems/setup-env.sh" >> ~/.bashrc

# Verify
rocprof-sys-sample --version
```

### Basic Usage

```bash
# Call-stack sampling (easiest, recommended)
rocprof-sys-sample -f 1000 -- ./app

# Binary instrumentation (more detailed)
rocprof-sys-instrument -o app.inst -- ./app
rocprof-sys-run -- ./app.inst

# Python profiling
rocprof-sys-python -- script.py
```

### Configuration

```bash
# Generate default config
rocprof-sys-avail -G rocprof-sys.cfg

# Edit config file
vim rocprof-sys.cfg

# Use custom config
export ROCPROFSYS_CONFIG_FILE=$(pwd)/rocprof-sys.cfg
rocprof-sys-run -- ./app
```

### Output

rocprofiler-systems generates `.proto` files - view them in Perfetto:
1. Go to https://ui.perfetto.dev
2. Click "Open trace file"
3. Select the `.proto` file
4. Explore timeline, GPU activity, and call stacks

## Tool 3: rocprofiler-compute (Kernel Analysis)

rocprofiler-compute (formerly Omniperf) provides detailed kernel analysis with comprehensive metrics - like NVIDIA Nsight Compute.

### Installation (No Sudo Required)

**Method 1: From Package (if available)**
```bash
# May still use old package name
sudo apt install omniperf

# Or from tarball (no sudo)
wget https://github.com/ROCm/omniperf/releases/download/v2.0.1/omniperf-v2.0.1-Linux.tar.gz
mkdir -p ~/.local/rocprofiler-compute
tar -xzf omniperf-*.tar.gz -C ~/.local/rocprofiler-compute --strip-components=1
echo 'export PATH=~/.local/rocprofiler-compute/bin:${PATH}' >> ~/.bashrc
source ~/.bashrc
```

**Method 2: Build from Source**
```bash
# Clone rocm-systems
git clone https://github.com/ROCm/rocm-systems.git
cd rocm-systems/projects/rocprofiler-compute

# Install Python dependencies
python3 -m pip install --user -r requirements.txt

# Build and install
cmake -B build -DCMAKE_INSTALL_PREFIX=~/.local/rocprofiler-compute
cmake --build build -j$(nproc)
cmake --install build

# Add to PATH
echo 'export PATH=~/.local/rocprofiler-compute/bin:${PATH}' >> ~/.bashrc
source ~/.bashrc
```

### Basic Usage

```bash
# Profile application (collects all metrics)
rocprof-compute profile -n analysis -- ./app

# Analyze with web UI (opens browser)
rocprof-compute analyze -p workloads/analysis --gui

# Generate HTML report
rocprof-compute analyze -p workloads/analysis --report-type html -o report.html
```

**If using old command name:**
```bash
omniperf profile -n analysis -- ./app
omniperf analyze -p workloads/analysis --gui
```

### Profile Options

```bash
# Profile specific kernel
rocprof-compute profile -n kernel_study --kernel-names "matmul_kernel" -- ./app

# Roof-only mode (faster, just roofline analysis)
rocprof-compute profile -n roof --roof-only -- ./app

# Filter by kernel regex
rocprof-compute profile -n filtered --kernel-names "^matmul" -- ./app
```

### Analyze Results

```bash
# Web UI (interactive, recommended)
rocprof-compute analyze -p workloads/analysis --gui
# Opens at http://localhost:8050

# Command-line report
rocprof-compute analyze -p workloads/analysis

# HTML report (for sharing)
rocprof-compute analyze -p workloads/analysis --report-type html -o report.html

# List available metrics
rocprof-compute analyze --list-metrics
```

### Output

rocprofiler-compute generates:
- `workloads/<name>/` - Directory with profiling data
- Web UI with comprehensive metrics:
  - Roofline charts
  - Memory bandwidth analysis
  - Compute utilization
  - Cache hit rates
  - Instruction mix

### Key Metrics Categories

- **System Speed-of-Light**: How close to theoretical peak performance
- **Compute Units**: Wavefront occupancy, execution efficiency
- **Local Data Share (LDS)**: LDS bandwidth and conflicts
- **Instruction Mix**: Types of instructions executed
- **Memory**: L1/L2 cache hits, bandwidth utilization
- **Roofline Analysis**: Performance vs bandwidth/compute limits

## Quick Command Reference

### Get Basic Stats

```bash
# rocprof (always available)
rocprof --stats ./app
```

### Generate Timeline

```bash
# rocprof (basic)
rocprof --sys-trace --hip-trace ./app
# View results.json in chrome://tracing

# rocprofiler-systems (advanced)
rocprof-sys-sample -f 1000 -- ./app
# View .proto at https://ui.perfetto.dev
```

### Analyze Kernels

```bash
# rocprof (basic metrics)
rocprof -i metrics.txt ./app
cat results.csv

# rocprofiler-compute (comprehensive analysis)
rocprof-compute profile -n analysis -- ./app
rocprof-compute analyze -p workloads/analysis --gui
```

### Profile Python/PyTorch

```bash
# rocprof (basic)
rocprof --hip-trace python train.py

# rocprofiler-systems (advanced)
rocprof-sys-python -- train.py
```

## Profiling Workflow

### Step 1: Quick Stats (rocprof)

```bash
# Start with rocprof for quick overview
rocprof --stats ./app

# Identify hot kernels from results.stats.csv
cat results.stats.csv
```

### Step 2: Timeline Analysis (rocprofiler-systems)

```bash
# Use rocprofiler-systems for detailed timeline (if installed)
rocprof-sys-sample -f 1000 -- ./app
# View in Perfetto

# OR use rocprof for basic timeline
rocprof --sys-trace --hip-trace ./app
# View in Chrome Tracing
```

### Step 3: Kernel Optimization (rocprofiler-compute)

```bash
# Deep dive into hot kernels with rocprofiler-compute
rocprof-compute profile -n kernel_analysis --kernel-names "hot_kernel" -- ./app
rocprof-compute analyze -p workloads/kernel_analysis --gui

# OR collect specific metrics with rocprof
cat > metrics.txt << 'EOF'
pmc: SQ_WAVES
pmc: SQ_INSTS_VALU
pmc: TCC_HIT
pmc: TCC_MISS
EOF

rocprof --kernel-name "hot_kernel" -i metrics.txt ./app
cat results.csv
```

## Examples with This Repository

### Build AMD Examples

```bash
# Use the build script
./build_amd.sh

# Or manually
cmake -S . -B build_amd -DUSE_AMD=ON -DENABLE_LINEINFO=ON
cmake --build build_amd -j
```

### Profile Matrix Multiplication

```bash
# Quick stats
rocprof --stats ./build_amd/matmul_hip

# Timeline
rocprof --sys-trace --hip-trace ./build_amd/matmul_hip

# Or with rocprofiler-systems
rocprof-sys-sample -f 1000 -- ./build_amd/matmul_hip
```

### Compare HIP vs rocBLAS

```bash
# Profile naive implementation
rocprof --stats -o naive ./build_amd/matmul_hip

# Profile optimized with rocBLAS
rocprof --stats -o rocblas ./build_amd/matmul_rocblas

# Compare
diff naive_results.stats.csv rocblas_results.stats.csv
```

## Common Options

### rocprof Options

```bash
--stats                    # Print kernel statistics
--sys-trace                # System-level trace
--hip-trace                # HIP API trace
--hsa-trace                # HSA API trace
-i metrics.txt             # Collect specific metrics
--kernel-name "name"       # Filter by kernel name
-o prefix                  # Output file prefix
-d directory               # Output directory
--timestamp on             # Add timestamps
```

### rocprofiler-systems Options

```bash
rocprof-sys-sample -f FREQ         # Sampling frequency (Hz)
rocprof-sys-run --trace            # Enable tracing
rocprof-sys-instrument -o OUT      # Output instrumented binary
rocprof-sys-python                 # Profile Python
```

### rocprofiler-compute Options

```bash
rocprof-compute profile -n NAME              # Profile with name
rocprof-compute profile --kernel-names "regex"  # Filter kernels
rocprof-compute profile --roof-only          # Roofline only (faster)
rocprof-compute analyze -p PATH --gui        # Analyze with web UI
rocprof-compute analyze --list-metrics       # List available metrics
```

## Environment Variables

### rocprof

```bash
# Output directory
export ROCPROF_OUTPUT_DIR=./my_results

# Enable HIP tracing
export HIP_TRACE_API=1
```

### rocprofiler-systems

```bash
# Config file location
export ROCPROFSYS_CONFIG_FILE=/path/to/config.cfg

# Output path
export ROCPROFSYS_OUTPUT_PATH=./profile_results
```

## Troubleshooting

### rocprof: command not found

```bash
# Add ROCm to PATH
export PATH=/opt/rocm/bin:${PATH}
echo 'export PATH=/opt/rocm/bin:${PATH}' >> ~/.bashrc
```

### rocprof-sys-sample: command not found

You need to install rocprofiler-systems (see installation section above).

### rocprof-compute: command not found

You need to install rocprofiler-compute (see installation section above). You can also try the old command name:
```bash
which omniperf  # May still be available as omniperf
```

### Cannot write to output directory

```bash
# Specify writable directory
rocprof -d ~/profiling_output --stats ./app
```

### Large trace files

```bash
# Filter by kernel
rocprof --kernel-name "specific_kernel" ./app

# Limit execution time
timeout 30s rocprof --hip-trace ./app
```

## Viewing Results

### Chrome Tracing

1. Open Chrome browser
2. Go to `chrome://tracing`
3. Click "Load"
4. Select `results.json` file

### Perfetto (Recommended)

1. Go to https://ui.perfetto.dev
2. Click "Open trace file"
3. Select `.json` or `.proto` file
4. Use Perfetto's powerful analysis tools

### CSV Files

```bash
# View in terminal
cat results.stats.csv
cat results.hip_stats.csv

# Open in spreadsheet
libreoffice results.stats.csv
```

## Tips

1. **Always build with line info** for better profiling:
   ```bash
   cmake -DENABLE_LINEINFO=ON
   ```

2. **Start simple**: Use `rocprof --stats` first, then add timeline/metrics

3. **Filter kernels** to reduce overhead:
   ```bash
   rocprof --kernel-name "my_kernel" ./app
   ```

4. **Use rocprofiler-systems for Python** - it has better Python support

5. **View in Perfetto** for best visualization experience

## Tool Comparison Summary

| Feature | rocprof | rocprofiler-systems | rocprofiler-compute |
|---------|---------|---------------------|---------------------|
| **Installation** | ✅ With ROCm | Separate | Separate |
| **Purpose** | Basic profiling | Timeline/system | Kernel analysis |
| **Timeline trace** | ✅ Basic | ✅ Advanced | ❌ |
| **Kernel metrics** | ✅ Basic | ❌ | ✅ Comprehensive |
| **Roofline** | ❌ | ❌ | ✅ |
| **Python support** | ✅ CLI | ✅ Native | ✅ |
| **Web UI** | ❌ | Perfetto | ✅ Built-in |
| **Output** | JSON, CSV | .proto | Web dashboard |
| **Like NVIDIA** | Basic | Nsight Systems | Nsight Compute |
| **When to use** | Quick checks | Understand flow | Optimize kernels |

## Resources

- **rocm-systems repository**: https://github.com/ROCm/rocm-systems
- **rocprof**: Included with ROCm at `/opt/rocm/bin/rocprof`
- **rocprofiler-systems**: https://github.com/ROCm/rocprofiler-systems
- **rocprofiler-compute**: In rocm-systems/projects/rocprofiler-compute
- **Perfetto UI**: https://ui.perfetto.dev
- **ROCm Documentation**: https://rocm.docs.amd.com/

## Quick Start Checklist

- [ ] Verify rocprof is available: `which rocprof`
- [ ] Build examples with line info: `./build_amd.sh`
- [ ] Get quick stats: `rocprof --stats ./build_amd/matmul_hip`
- [ ] Generate timeline: `rocprof --sys-trace --hip-trace ./build_amd/matmul_hip`
- [ ] View in Perfetto: https://ui.perfetto.dev
- [ ] (Optional) Install rocprofiler-systems for advanced timeline profiling
- [ ] (Optional) Install rocprofiler-compute for deep kernel analysis
