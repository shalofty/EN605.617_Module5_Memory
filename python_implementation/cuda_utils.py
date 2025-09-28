"""
CUDA Memory Assignment Utilities
Module 5 - Memory Assignment Utility Functions

This file contains utility functions for timing, performance analysis,
and common operations used across all memory type implementations.
"""

import time
import cupy as cp
import numpy as np
from typing import Tuple, List, Dict, Any, Callable
from cuda_config import (
    TIMING_ITERATIONS, WARMUP_ITERATIONS, PERFORMANCE_THRESHOLD_MS,
    ERROR_MESSAGES, SUCCESS_MESSAGES
)


def time_kernel_execution(kernel_func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """
    Time kernel execution with warmup and multiple iterations for accuracy.
    
    Args:
        kernel_func: The kernel function to time
        *args: Arguments to pass to the kernel function
        **kwargs: Keyword arguments to pass to the kernel function
        
    Returns:
        Tuple of (result, average_execution_time_ms)
    """
    # Warmup iterations
    for _ in range(WARMUP_ITERATIONS):
        kernel_func(*args, **kwargs)
    
    # Synchronize before timing
    cp.cuda.Stream.null.synchronize()
    
    # Time multiple iterations
    times = []
    for _ in range(TIMING_ITERATIONS):
        start_time = time.perf_counter()
        result = kernel_func(*args, **kwargs)
        cp.cuda.Stream.null.synchronize()
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000)  # Convert to ms
    
    avg_time = sum(times) / len(times)
    return result, avg_time


def calculate_memory_bandwidth(data_size_bytes: int, execution_time_ms: float) -> float:
    """
    Calculate memory bandwidth in GB/s.
    
    Args:
        data_size_bytes: Size of data transferred in bytes
        execution_time_ms: Execution time in milliseconds
        
    Returns:
        Memory bandwidth in GB/s
    """
    if execution_time_ms == 0:
        return 0.0
    
    # Convert bytes to GB and ms to seconds
    data_size_gb = data_size_bytes / (1024**3)
    execution_time_s = execution_time_ms / 1000.0
    
    # Bandwidth = data_size / time (accounting for read+write)
    bandwidth_gbps = (data_size_gb * 2) / execution_time_s
    return bandwidth_gbps


def validate_thread_configuration(thread_count: int, block_size: int) -> bool:
    """
    Validate thread and block configuration.
    
    Args:
        thread_count: Number of threads
        block_size: Size of each block
        
    Returns:
        True if configuration is valid, False otherwise
    """
    if thread_count < 64:
        print(f"Error: {ERROR_MESSAGES['invalid_thread_count']}")
        return False
    
    if block_size <= 0 or block_size > 1024:
        print(f"Error: {ERROR_MESSAGES['invalid_block_size']}")
        return False
    
    return True


def create_test_data(size: int, data_type: str = 'float32') -> Tuple[np.ndarray, cp.ndarray]:
    """
    Create test data arrays for host and device.
    
    Args:
        size: Size of the data array
        data_type: Data type for the arrays
        
    Returns:
        Tuple of (host_array, device_array)
    """
    # Create random test data
    host_data = np.random.rand(size).astype(data_type)
    device_data = cp.asarray(host_data)
    
    return host_data, device_data


def verify_results(host_result: np.ndarray, device_result: cp.ndarray, 
                  tolerance: float = 1e-6) -> bool:
    """
    Verify that host and device results match within tolerance.
    
    Args:
        host_result: Result from host computation
        device_result: Result from device computation
        tolerance: Tolerance for comparison
        
    Returns:
        True if results match, False otherwise
    """
    # Convert device result to numpy for comparison
    device_result_cpu = cp.asnumpy(device_result)
    
    # Check if arrays are close
    return np.allclose(host_result, device_result_cpu, atol=tolerance)


def print_performance_summary(memory_type: str, data_size: int, 
                            thread_count: int, block_size: int,
                            execution_time_ms: float, bandwidth_gbps: float):
    """
    Print formatted performance summary.
    
    Args:
        memory_type: Type of memory used
        data_size: Size of data processed
        thread_count: Number of threads used
        block_size: Block size used
        execution_time_ms: Execution time in milliseconds
        bandwidth_gbps: Memory bandwidth in GB/s
    """
    print(f"\n{'='*60}")
    print(f"PERFORMANCE SUMMARY - {memory_type.upper()} MEMORY")
    print(f"{'='*60}")
    print(f"Data Size: {data_size:,} elements")
    print(f"Thread Count: {thread_count:,}")
    print(f"Block Size: {block_size}")
    print(f"Execution Time: {execution_time_ms:.4f} ms")
    print(f"Memory Bandwidth: {bandwidth_gbps:.2f} GB/s")
    print(f"{'='*60}")


def create_performance_chart(results: Dict[str, List[float]], 
                           title: str = "Performance Comparison"):
    """
    Create performance comparison chart.
    
    Args:
        results: Dictionary of memory types and their execution times
        title: Chart title
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 8))
    
    for memory_type, times in results.items():
        plt.plot(times, label=memory_type.title(), marker='o')
    
    plt.xlabel('Data Size (elements)')
    plt.ylabel('Execution Time (ms)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.show()


def log_execution_details(memory_type: str, config: Dict[str, Any], 
                         performance: Dict[str, float]):
    """
    Log detailed execution information.
    
    Args:
        memory_type: Type of memory used
        config: Configuration parameters
        performance: Performance metrics
    """
    print(f"\nExecution Details for {memory_type.upper()} Memory:")
    print(f"Configuration: {config}")
    print(f"Performance: {performance}")
    
    # Log to file if needed
    with open('execution_log.txt', 'a') as f:
        f.write(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')} - {memory_type}\n")
        f.write(f"Config: {config}\n")
        f.write(f"Performance: {performance}\n")


def cleanup_memory():
    """
    Clean up GPU memory and reset CUDA context.
    """
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    print("GPU memory cleaned up successfully")


def check_cuda_availability() -> bool:
    """
    Check if CUDA is available and properly configured.
    
    Returns:
        True if CUDA is available, False otherwise
    """
    try:
        if not cp.cuda.is_available():
            print(f"Error: {ERROR_MESSAGES['cuda_not_available']}")
            return False
        
        # Test basic CUDA operations
        test_array = cp.array([1, 2, 3, 4, 5])
        result = cp.sum(test_array)
        
        if result == 15:  # Sum of [1,2,3,4,5]
            print(f"Success: {SUCCESS_MESSAGES['setup_complete']}")
            return True
        else:
            print("Error: CUDA test failed")
            return False
            
    except Exception as e:
        print(f"Error: CUDA availability check failed: {e}")
        return False
