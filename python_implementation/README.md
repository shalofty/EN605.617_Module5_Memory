# Python/CuPy Implementation

This directory contains the complete Python/CuPy implementation of the CUDA Memory Assignment.

## Files

- **`cuda_memory_assignment.ipynb`** - Main Jupyter notebook with all implementations
- **`cuda_cli.py`** - Command line interface
- **`cuda_config.py`** - Configuration parameters
- **`cuda_utils.py`** - Utility functions
- **`test_cuda_memory.py`** - Standalone testing script
- **`run_memory_tests.py`** - Comprehensive testing integration
- **`run.sh`** - Execution script

## Quick Start

### Option 1: Jupyter Notebook
```bash
# Open in Jupyter/Google Colab
jupyter notebook cuda_memory_assignment.ipynb
```

### Option 2: Command Line
```bash
# Test all memory types
python3 test_cuda_memory.py --memory-type all

# Test specific memory type
python3 test_cuda_memory.py --memory-type global --threads 512 --blocks 128

# Run comprehensive testing
python3 run_memory_tests.py --comprehensive
```

### Option 3: Run Script
```bash
# Make executable and run
chmod +x run.sh
./run.sh --memory all --verbose
```

## Features

- **High-level abstraction** using CuPy for CUDA operations
- **Easy to understand** and modify
- **Comprehensive testing** with multiple configurations
- **Performance analysis** with detailed metrics
- **Command line interface** with all options
- **Jupyter notebook** for interactive development

## Requirements

- Python 3.8+
- CuPy (CUDA-enabled NumPy)
- NumPy
- Matplotlib

## Installation

```bash
pip install cupy-cuda12x numpy matplotlib
```

## Testing

```bash
# Basic test
python3 test_cuda_memory.py --memory-type host --data-size small

# All memory types
python3 test_cuda_memory.py --memory-type all --data-size medium

# Comprehensive testing
python3 run_memory_tests.py --comprehensive
```

## Performance

The Python implementation provides excellent performance analysis and is perfect for:
- Learning CUDA concepts
- Rapid prototyping
- Educational purposes
- Cross-platform development

See the main repository README for detailed performance comparisons.
