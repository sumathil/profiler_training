#!/bin/bash

# profile_all.sh - Automated profiling script for AMD examples
# This script runs multiple profiling tools on the example applications
# and generates a summary report

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
OUTPUT_DIR="profiling_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="${OUTPUT_DIR}/report_${TIMESTAMP}.txt"

# Check if ROCm tools are available
check_tools() {
    echo -e "${BLUE}Checking AMD profiling tools...${NC}"

    local tools=("rocprof" "rocminfo" "hipcc")
    local missing_tools=()

    for tool in "${tools[@]}"; do
        if command -v "$tool" &> /dev/null; then
            echo -e "${GREEN}✓${NC} $tool found"
        else
            echo -e "${RED}✗${NC} $tool not found"
            missing_tools+=("$tool")
        fi
    done

    # Check for Chrome/Chromium for timeline viewing
    if command -v google-chrome &> /dev/null || command -v chromium &> /dev/null; then
        echo -e "${GREEN}✓${NC} Chrome/Chromium found (for timeline viewing)"
    else
        echo -e "${YELLOW}!${NC} Chrome/Chromium not found (optional, for viewing timeline traces)"
    fi

    if [ ${#missing_tools[@]} -ne 0 ]; then
        echo -e "${RED}Error: Missing required tools: ${missing_tools[*]}${NC}"
        echo "Please install ROCm: https://rocm.docs.amd.com/"
        exit 1
    fi

    echo ""
}

# Create output directory
setup_output_dir() {
    mkdir -p "$OUTPUT_DIR"
    echo -e "${BLUE}Output directory: $OUTPUT_DIR${NC}\n"
}

# Profile with rocprof (basic stats)
profile_rocprof_stats() {
    local app=$1
    local name=$(basename "$app")

    echo -e "${BLUE}[rocprof] Profiling $name with basic stats...${NC}"

    local output_prefix="${OUTPUT_DIR}/${name}_rocprof_stats"

    rocprof --stats -o "${output_prefix}.csv" "$app" > /dev/null 2>&1

    if [ -f "${output_prefix}.stats.csv" ]; then
        echo -e "${GREEN}✓${NC} Stats saved to ${output_prefix}.stats.csv"

        # Extract key metrics
        echo "  Kernels found:"
        tail -n +2 "${output_prefix}.stats.csv" | awk -F',' '{printf "    - %s: %.2f ms\n", $1, $3/1000000}'
    else
        echo -e "${RED}✗${NC} Failed to generate stats"
    fi

    echo ""
}

# Profile with rocprof (detailed metrics)
profile_rocprof_metrics() {
    local app=$1
    local name=$(basename "$app")

    if [ ! -f "metrics.txt" ]; then
        echo -e "${YELLOW}[rocprof] Skipping metrics (metrics.txt not found)${NC}\n"
        return
    fi

    echo -e "${BLUE}[rocprof] Profiling $name with detailed metrics...${NC}"

    local output_prefix="${OUTPUT_DIR}/${name}_rocprof_metrics"

    rocprof -i metrics.txt -o "${output_prefix}.csv" "$app" > /dev/null 2>&1

    if [ -f "${output_prefix}.csv" ]; then
        echo -e "${GREEN}✓${NC} Metrics saved to ${output_prefix}.csv"

        # Show sample metrics if available
        if command -v head &> /dev/null && command -v column &> /dev/null; then
            echo "  Sample metrics:"
            head -n 2 "${output_prefix}.csv" | column -t -s',' | sed 's/^/    /'
        fi
    else
        echo -e "${RED}✗${NC} Failed to generate metrics"
    fi

    echo ""
}

# Profile with rocprof (system trace)
profile_rocprof_trace() {
    local app=$1
    local name=$(basename "$app")

    echo -e "${BLUE}[rocprof] Profiling $name with system trace...${NC}"

    local output_prefix="${OUTPUT_DIR}/${name}_rocprof_trace"

    rocprof --sys-trace --hip-trace --hsa-trace -o "${output_prefix}" "$app" > /dev/null 2>&1

    if [ -f "${output_prefix}.json" ]; then
        echo -e "${GREEN}✓${NC} Timeline trace saved to ${output_prefix}.json"
        echo "  View in Chrome: chrome://tracing"
    elif [ -f "results.json" ]; then
        mv results.json "${output_prefix}.json"
        echo -e "${GREEN}✓${NC} Timeline trace saved to ${output_prefix}.json"
        echo "  View in Chrome: chrome://tracing"
    else
        echo -e "${YELLOW}!${NC} Timeline trace generation failed"
    fi

    echo ""
}


# Generate summary report
generate_report() {
    echo -e "${BLUE}Generating summary report...${NC}"

    {
        echo "========================================"
        echo "AMD Profiling Report"
        echo "========================================"
        echo "Generated: $(date)"
        echo ""

        echo "System Information:"
        echo "-------------------"
        rocminfo | grep -E "Name:|Marketing Name:|Compute Unit:" | head -n 6
        echo ""

        echo "Profiled Applications:"
        echo "---------------------"

        # List all profiled stats
        for stats_file in "${OUTPUT_DIR}"/*_rocprof_stats.stats.csv; do
            if [ -f "$stats_file" ]; then
                local app_name=$(basename "$stats_file" "_rocprof_stats.stats.csv")
                echo ""
                echo "Application: $app_name"
                echo "Kernels:"
                tail -n +2 "$stats_file" | awk -F',' '{printf "  %-40s %10.2f ms  (%5.1f%%)\n", $1, $3/1000000, $5}'
            fi
        done

        echo ""
        echo "Output Files:"
        echo "-------------"
        ls -lh "$OUTPUT_DIR" | tail -n +2 | awk '{printf "  %s  %s\n", $5, $9}'

        echo ""
        echo "========================================"

    } | tee "$REPORT_FILE"

    echo -e "${GREEN}✓${NC} Report saved to $REPORT_FILE\n"
}

# Main profiling function
profile_application() {
    local app=$1

    if [ ! -f "$app" ]; then
        echo -e "${RED}Error: Application not found: $app${NC}"
        return 1
    fi

    if [ ! -x "$app" ]; then
        echo -e "${RED}Error: Application not executable: $app${NC}"
        return 1
    fi

    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}Profiling: $(basename $app)${NC}"
    echo -e "${GREEN}============================================${NC}\n"

    # Run different profiling tools
    profile_rocprof_stats "$app"
    profile_rocprof_metrics "$app"
    profile_rocprof_trace "$app"

    echo -e "${GREEN}============================================${NC}\n"
}

# Main execution
main() {
    echo -e "${GREEN}"
    echo "========================================"
    echo "AMD Profiling Automation Script"
    echo "========================================"
    echo -e "${NC}\n"

    check_tools
    setup_output_dir

    # Find built applications
    local apps=()

    if [ -d "build" ]; then
        while IFS= read -r app; do
            apps+=("$app")
        done < <(find build -maxdepth 1 -type f -executable ! -name "*.so" ! -name "*.a")
    fi

    if [ ${#apps[@]} -eq 0 ]; then
        echo -e "${RED}Error: No built applications found in build/${NC}"
        echo "Please build the examples first:"
        echo "  mkdir build && cd build"
        echo "  cmake .. && cmake --build ."
        exit 1
    fi

    echo -e "${BLUE}Found ${#apps[@]} application(s) to profile${NC}\n"

    # Profile each application
    for app in "${apps[@]}"; do
        profile_application "$app"
    done

    # Generate summary report
    generate_report

    echo -e "${GREEN}Profiling complete!${NC}"
    echo -e "Results saved in: ${BLUE}$OUTPUT_DIR${NC}\n"

    # Print helpful next steps
    echo -e "${YELLOW}Next steps:${NC}"
    echo "  1. Review the report: cat $REPORT_FILE"
    echo "  2. Examine CSV files for detailed metrics"
    echo "  3. View timeline traces in Chrome:"
    echo "     - Open Chrome browser"
    echo "     - Navigate to chrome://tracing"
    echo "     - Load *_trace.json files from $OUTPUT_DIR"
    echo "  4. Analyze hardware metrics in *_metrics.csv files"
}

# Run main function
main "$@"
