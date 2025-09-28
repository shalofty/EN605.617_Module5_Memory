"""
CUDA Memory Assignment Configuration
Module 5 - Memory Assignment Configuration Parameters

This file contains all configuration constants and parameters
for the CUDA memory assignment implementation.
"""

# Data size configurations (minimum 64 threads as required)
DATA_SIZES = {
    'small': 64,      # Minimum required threads
    'medium': 256,    # 4x minimum
    'large': 1024,    # 16x minimum
    'xlarge': 4096    # 64x minimum for performance analysis
}

# Thread block configurations
BLOCK_SIZES = {
    'small': 32,      # 1 warp
    'medium': 64,     # 2 warps
    'large': 128,     # 4 warps
    'xlarge': 256     # 8 warps
}

# Thread count configurations (minimum 64 as required)
THREAD_COUNTS = {
    'min': 64,        # Minimum required
    'medium': 128,    # 2x minimum
    'large': 256,     # 4x minimum
    'xlarge': 512     # 8x minimum
}

# Memory type identifiers
MEMORY_TYPES = {
    'host': 'host',
    'global': 'global', 
    'shared': 'shared',
    'constant': 'constant',
    'register': 'register',
    'all': 'all'      # Run all memory types
}

# Performance measurement parameters
TIMING_ITERATIONS = 100  # Number of iterations for timing accuracy
WARMUP_ITERATIONS = 10   # Warmup iterations before timing

# CUDA kernel parameters
MAX_THREADS_PER_BLOCK = 1024  # Maximum threads per block
SHARED_MEMORY_SIZE = 48 * 1024  # 48KB shared memory limit

# Output formatting
LINE_WIDTH = 80  # Maximum line width for code formatting
FUNCTION_LINE_LIMIT = 40  # Maximum lines per function

# Performance analysis thresholds
PERFORMANCE_THRESHOLD_MS = 1.0  # Threshold for performance analysis
MEMORY_BANDWIDTH_THRESHOLD_GBPS = 100  # Memory bandwidth threshold

# Command line argument defaults
DEFAULT_THREADS = 256
DEFAULT_BLOCKS = 64
DEFAULT_DATA_SIZE = 'medium'
DEFAULT_MEMORY_TYPE = 'all'

# File paths
OUTPUT_DIR = 'output'
RESULTS_FILE = 'performance_results.txt'
TIMING_FILE = 'timing_data.csv'

# Error messages
ERROR_MESSAGES = {
    'cuda_not_available': 'CUDA is not available on this system',
    'invalid_thread_count': 'Thread count must be >= 64',
    'invalid_block_size': 'Block size must be > 0 and <= 1024',
    'invalid_data_size': 'Invalid data size specified',
    'invalid_memory_type': 'Invalid memory type specified',
    'memory_allocation_failed': 'Failed to allocate memory',
    'kernel_execution_failed': 'Kernel execution failed'
}

# Success messages
SUCCESS_MESSAGES = {
    'setup_complete': 'CUDA environment setup complete',
    'memory_test_passed': 'Memory test completed successfully',
    'performance_analysis_complete': 'Performance analysis complete',
    'all_tests_passed': 'All memory type tests passed'
}
