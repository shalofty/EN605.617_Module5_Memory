#!/bin/bash

# CUDA Memory Assignment - Run Script
# Module 5 - Memory Assignment Execution Script
# 
# This script provides easy execution of the CUDA memory assignment
# with various command line argument combinations.

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "CUDA Memory Assignment - Run Script"
    echo "===================================="
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -t, --threads N         Number of threads (default: 256)"
    echo "  -b, --blocks N          Block size (default: 64)"
    echo "  -s, --size SIZE         Data size: small, medium, large, xlarge (default: medium)"
    echo "  -m, --memory TYPE       Memory type: host, global, shared, constant, register, all (default: all)"
    echo "  -i, --iterations N      Number of timing iterations (default: 100)"
    echo "  -v, --verbose           Enable verbose output"
    echo "  -c, --compare           Compare performance across memory types"
    echo "  -a, --all-configs       Test all thread/block configurations"
    echo "  -g, --generate-charts   Generate performance charts"
    echo "  -o, --output FILE       Output file for results"
    echo ""
    echo "Examples:"
    echo "  $0 --memory global --threads 512"
    echo "  $0 --size large --compare --verbose"
    echo "  $0 --all-configs --generate-charts"
    echo ""
}

# Default values
THREADS=256
BLOCKS=64
SIZE="medium"
MEMORY="all"
ITERATIONS=100
VERBOSE=false
COMPARE=false
ALL_CONFIGS=false
GENERATE_CHARTS=false
OUTPUT_FILE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -t|--threads)
            THREADS="$2"
            shift 2
            ;;
        -b|--blocks)
            BLOCKS="$2"
            shift 2
            ;;
        -s|--size)
            SIZE="$2"
            shift 2
            ;;
        -m|--memory)
            MEMORY="$2"
            shift 2
            ;;
        -i|--iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -c|--compare)
            COMPARE=true
            shift
            ;;
        -a|--all-configs)
            ALL_CONFIGS=true
            shift
            ;;
        -g|--generate-charts)
            GENERATE_CHARTS=true
            shift
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate arguments
if [ "$THREADS" -lt 64 ]; then
    print_error "Thread count must be at least 64 (got $THREADS)"
    exit 1
fi

if [ "$BLOCKS" -le 0 ] || [ "$BLOCKS" -gt 1024 ]; then
    print_error "Block size must be between 1 and 1024 (got $BLOCKS)"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    print_error "Python3 is not installed or not in PATH"
    exit 1
fi

# Check if required files exist
if [ ! -f "cuda_cli.py" ]; then
    print_error "cuda_cli.py not found. Please run this script from the project directory."
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p output

# Build command
CMD="python3 cuda_cli.py --threads $THREADS --blocks $BLOCKS --data-size $SIZE --memory-type $MEMORY --iterations $ITERATIONS"

if [ "$VERBOSE" = true ]; then
    CMD="$CMD --verbose"
fi

if [ "$COMPARE" = true ]; then
    CMD="$CMD --compare-performance"
fi

if [ "$ALL_CONFIGS" = true ]; then
    CMD="$CMD --test-all-configs"
fi

if [ "$GENERATE_CHARTS" = true ]; then
    CMD="$CMD --generate-charts"
fi

if [ -n "$OUTPUT_FILE" ]; then
    CMD="$CMD --output-file $OUTPUT_FILE"
fi

# Print configuration
print_status "Starting CUDA Memory Assignment..."
echo "Configuration:"
echo "  Threads: $THREADS"
echo "  Blocks: $BLOCKS"
echo "  Data Size: $SIZE"
echo "  Memory Type: $MEMORY"
echo "  Iterations: $ITERATIONS"
echo "  Verbose: $VERBOSE"
echo "  Compare: $COMPARE"
echo "  All Configs: $ALL_CONFIGS"
echo "  Generate Charts: $GENERATE_CHARTS"
if [ -n "$OUTPUT_FILE" ]; then
    echo "  Output File: $OUTPUT_FILE"
fi
echo ""

# Execute the command
print_status "Executing: $CMD"
echo ""

if eval $CMD; then
    print_success "CUDA Memory Assignment completed successfully!"
    
    # Show output files if any were created
    if [ -d "output" ] && [ "$(ls -A output)" ]; then
        print_status "Output files created:"
        ls -la output/
    fi
    
    if [ -f "execution_log.txt" ]; then
        print_status "Execution log created: execution_log.txt"
    fi
    
else
    print_error "CUDA Memory Assignment failed!"
    exit 1
fi

print_success "Run script completed successfully!"
