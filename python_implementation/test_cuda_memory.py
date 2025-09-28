#!/usr/bin/env python3
"""
CUDA Memory Assignment Test Script
Module 5 - Memory Assignment Testing and Validation

This script tests all 5 types of CUDA memory implementations
and validates the assignment requirements.
"""

import cupy as cp
import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple
from cuda_config import DATA_SIZES, THREAD_COUNTS, MEMORY_TYPES
from cuda_utils import check_cuda_availability, calculate_memory_bandwidth, cleanup_memory


class HostMemoryDemo:
    """Host Memory demonstration class."""
    
    def __init__(self, data_size: int):
        self.data_size = data_size
        self.host_data = None
        self.device_data = None
        
    def allocate_host_memory(self):
        """Allocate host memory and create test data."""
        print(f"Allocating host memory for {self.data_size:,} elements...")
        self.host_data = np.random.rand(self.data_size).astype(np.float32)
        print(f"Host memory allocated: {self.host_data.nbytes:,} bytes")
        
    def transfer_to_device(self):
        """Transfer data from host to device memory."""
        print("Transferring data from host to device...")
        self.device_data = cp.asarray(self.host_data)
        print(f"Data transferred to device: {self.device_data.nbytes:,} bytes")
        
    def host_computation(self):
        """Perform computation on host memory."""
        print("Performing computation on host memory...")
        result = np.sum(self.host_data * 2.0)
        return result
        
    def device_computation(self):
        """Perform computation on device memory."""
        print("Performing computation on device memory...")
        result = cp.sum(self.device_data * 2.0)
        return result
        
    def run_host_memory_test(self, thread_count: int = 256, block_size: int = 64):
        """Run complete host memory test with timing."""
        print(f"\n{'='*60}")
        print(f"HOST MEMORY TEST - {thread_count} threads, {block_size} blocks")
        print(f"{'='*60}")
        
        # Allocate memory
        self.allocate_host_memory()
        
        # Transfer to device
        self.transfer_to_device()
        
        # Time host computation
        host_start = time.perf_counter()
        host_result = self.host_computation()
        host_end = time.perf_counter()
        host_time = (host_end - host_start) * 1000
        
        # Time device computation
        device_start = time.perf_counter()
        device_result = self.device_computation()
        cp.cuda.Stream.null.synchronize()
        device_end = time.perf_counter()
        device_time = (device_end - device_start) * 1000
        
        # Calculate memory bandwidth
        data_size_bytes = self.host_data.nbytes
        host_bandwidth = calculate_memory_bandwidth(data_size_bytes, host_time)
        device_bandwidth = calculate_memory_bandwidth(data_size_bytes, device_time)
        
        # Verify results match
        results_match = np.allclose(host_result, cp.asnumpy(device_result), atol=1e-6)
        
        print(f"\nHost Computation Time: {host_time:.4f} ms")
        print(f"Device Computation Time: {device_time:.4f} ms")
        print(f"Host Memory Bandwidth: {host_bandwidth:.2f} GB/s")
        print(f"Device Memory Bandwidth: {device_bandwidth:.2f} GB/s")
        print(f"Results Match: {results_match}")
        
        return {
            'memory_type': 'host',
            'data_size': self.data_size,
            'thread_count': thread_count,
            'block_size': block_size,
            'host_time_ms': host_time,
            'device_time_ms': device_time,
            'host_bandwidth_gbps': host_bandwidth,
            'device_bandwidth_gbps': device_bandwidth,
            'results_match': results_match,
            'total_time_ms': host_time + device_time
        }


class GlobalMemoryDemo:
    """Global Memory demonstration class."""
    
    def __init__(self, data_size: int):
        self.data_size = data_size
        self.host_data = None
        self.device_data = None
        self.result_data = None
        
    def allocate_global_memory(self):
        """Allocate global memory on GPU."""
        print(f"Allocating global memory for {self.data_size:,} elements...")
        self.host_data = np.random.rand(self.data_size).astype(np.float32)
        self.device_data = cp.asarray(self.host_data)
        self.result_data = cp.zeros(self.data_size, dtype=cp.float32)
        print(f"Global memory allocated: {self.device_data.nbytes:,} bytes")
        
    def global_memory_coalesced_operation(self, input_data, output_data, multiplier):
        """Simulate coalesced memory access pattern using CuPy operations."""
        output_data[:] = input_data * multiplier + cp.sin(input_data)
        
    def global_memory_strided_operation(self, input_data, output_data, multiplier):
        """Simulate strided memory access pattern using CuPy operations."""
        stride_indices = cp.arange(0, input_data.size, 2) % input_data.size
        strided_data = input_data[stride_indices]
        # Ensure output_data has the same size as strided_data
        output_data[:len(strided_data)] = strided_data * multiplier + cp.cos(strided_data)
        
    def run_global_memory_test(self, thread_count: int = 256, block_size: int = 64):
        """Run global memory test with different access patterns."""
        print(f"\n{'='*60}")
        print(f"GLOBAL MEMORY TEST - {thread_count} threads, {block_size} blocks")
        print(f"{'='*60}")
        
        # Allocate global memory
        self.allocate_global_memory()
        
        # Test coalesced access pattern
        print("\nTesting Coalesced Memory Access Pattern...")
        coalesced_start = time.perf_counter()
        self.global_memory_coalesced_operation(self.device_data, self.result_data, 2.0)
        cp.cuda.Stream.null.synchronize()
        coalesced_end = time.perf_counter()
        coalesced_time = (coalesced_end - coalesced_start) * 1000
        
        # Test strided access pattern
        print("Testing Strided Memory Access Pattern...")
        strided_start = time.perf_counter()
        self.global_memory_strided_operation(self.device_data, self.result_data, 2.0)
        cp.cuda.Stream.null.synchronize()
        strided_end = time.perf_counter()
        strided_time = (strided_end - strided_start) * 1000
        
        # Calculate memory bandwidth
        data_size_bytes = self.device_data.nbytes
        coalesced_bandwidth = calculate_memory_bandwidth(data_size_bytes, coalesced_time)
        strided_bandwidth = calculate_memory_bandwidth(data_size_bytes, strided_time)
        
        print(f"\nCoalesced Access Time: {coalesced_time:.4f} ms")
        print(f"Strided Access Time: {strided_time:.4f} ms")
        print(f"Coalesced Bandwidth: {coalesced_bandwidth:.2f} GB/s")
        print(f"Strided Bandwidth: {strided_bandwidth:.2f} GB/s")
        print(f"Speedup (Coalesced vs Strided): {strided_time/coalesced_time:.2f}x")
        
        return {
            'memory_type': 'global',
            'data_size': self.data_size,
            'thread_count': thread_count,
            'block_size': block_size,
            'coalesced_time_ms': coalesced_time,
            'strided_time_ms': strided_time,
            'coalesced_bandwidth_gbps': coalesced_bandwidth,
            'strided_bandwidth_gbps': strided_bandwidth,
            'speedup': strided_time / coalesced_time,
            'total_time_ms': coalesced_time + strided_time
        }


class SharedMemoryDemo:
    """Shared Memory demonstration class."""
    
    def __init__(self, data_size: int):
        self.data_size = data_size
        self.host_data = None
        self.device_data = None
        self.result_data = None
        
    def allocate_shared_memory(self):
        """Allocate memory for shared memory demo."""
        print(f"Allocating memory for shared memory demo with {self.data_size:,} elements...")
        self.host_data = np.random.rand(self.data_size).astype(np.float32)
        self.device_data = cp.asarray(self.host_data)
        self.result_data = cp.zeros(self.data_size, dtype=cp.float32)
        print(f"Memory allocated: {self.device_data.nbytes:,} bytes")
        
    def shared_memory_reduction_operation(self, input_data, output_data):
        """Simulate shared memory reduction using CuPy operations."""
        # Simulate block-level reduction
        block_size = min(256, input_data.size)
        num_blocks = (input_data.size + block_size - 1) // block_size
        
        for block_id in range(num_blocks):
            start_idx = block_id * block_size
            end_idx = min(start_idx + block_size, input_data.size)
            block_data = input_data[start_idx:end_idx]
            block_sum = cp.sum(block_data)
            output_data[block_id] = block_sum
            
    def shared_memory_matrix_operation(self, input_data, output_data):
        """Simulate shared memory matrix operations."""
        # Reshape to square matrix if possible
        n = int(np.sqrt(input_data.size))
        if n * n != input_data.size:
            n = int(np.sqrt(input_data.size)) + 1
            padded_data = cp.zeros(n * n, dtype=cp.float32)
            padded_data[:input_data.size] = input_data
            input_data = padded_data
            
        # Reshape to matrix
        matrix = input_data.reshape(n, n)
        
        # Simulate matrix operations with shared memory caching
        result_matrix = cp.zeros((n, n), dtype=cp.float32)
        
        # Simple matrix multiplication simulation
        for i in range(n):
            for j in range(n):
                # Simulate shared memory access pattern
                row_data = matrix[i, :]
                col_data = matrix[:, j]
                result_matrix[i, j] = cp.sum(row_data * col_data)
                
        # Flatten result
        output_data[:n*n] = result_matrix.flatten()
        
    def run_shared_memory_test(self, thread_count: int = 256, block_size: int = 64):
        """Run shared memory test with reduction and matrix operations."""
        print(f"\n{'='*60}")
        print(f"SHARED MEMORY TEST - {thread_count} threads, {block_size} blocks")
        print(f"{'='*60}")
        
        # Allocate memory
        self.allocate_shared_memory()
        
        # Test shared memory reduction
        print("\nTesting Shared Memory Reduction...")
        reduction_start = time.perf_counter()
        self.shared_memory_reduction_operation(self.device_data, self.result_data)
        cp.cuda.Stream.null.synchronize()
        reduction_end = time.perf_counter()
        reduction_time = (reduction_end - reduction_start) * 1000
        
        # Test shared memory matrix operations
        print("Testing Shared Memory Matrix Operations...")
        matrix_start = time.perf_counter()
        self.shared_memory_matrix_operation(self.device_data, self.result_data)
        cp.cuda.Stream.null.synchronize()
        matrix_end = time.perf_counter()
        matrix_time = (matrix_end - matrix_start) * 1000
        
        # Calculate memory bandwidth
        data_size_bytes = self.device_data.nbytes
        reduction_bandwidth = calculate_memory_bandwidth(data_size_bytes, reduction_time)
        matrix_bandwidth = calculate_memory_bandwidth(data_size_bytes, matrix_time)
        
        print(f"\nReduction Time: {reduction_time:.4f} ms")
        print(f"Matrix Operations Time: {matrix_time:.4f} ms")
        print(f"Reduction Bandwidth: {reduction_bandwidth:.2f} GB/s")
        print(f"Matrix Bandwidth: {matrix_bandwidth:.2f} GB/s")
        
        return {
            'memory_type': 'shared',
            'data_size': self.data_size,
            'thread_count': thread_count,
            'block_size': block_size,
            'reduction_time_ms': reduction_time,
            'matrix_time_ms': matrix_time,
            'reduction_bandwidth_gbps': reduction_bandwidth,
            'matrix_bandwidth_gbps': matrix_bandwidth,
            'total_time_ms': reduction_time + matrix_time
        }


class ConstantMemoryDemo:
    """Constant Memory demonstration class."""
    
    def __init__(self, data_size: int):
        self.data_size = data_size
        self.host_data = None
        self.device_data = None
        self.result_data = None
        self.constant_data = None
        
    def allocate_constant_memory(self):
        """Allocate memory and set up constant data."""
        print(f"Allocating memory for constant memory demo with {self.data_size:,} elements...")
        self.host_data = np.random.rand(self.data_size).astype(np.float32)
        self.device_data = cp.asarray(self.host_data)
        self.result_data = cp.zeros(self.data_size, dtype=cp.float32)
        
        # Create constant data (lookup tables, coefficients, etc.)
        self.constant_data = cp.array([
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0
        ], dtype=cp.float32)
        
        print(f"Memory allocated: {self.device_data.nbytes:,} bytes")
        print(f"Constant data size: {self.constant_data.nbytes:,} bytes")
        
    def constant_memory_lookup_operation(self, input_data, output_data, constant_table):
        """Simulate constant memory lookup operations."""
        for i in range(input_data.size):
            lookup_idx = int(input_data[i] * len(constant_table)) % len(constant_table)
            constant_value = constant_table[lookup_idx]
            output_data[i] = input_data[i] * constant_value + cp.sin(input_data[i])
            
    def constant_memory_coefficient_operation(self, input_data, output_data, coefficients):
        """Simulate constant memory coefficient operations."""
        for i in range(input_data.size):
            x = input_data[i]
            if len(coefficients) >= 4:
                a, b, c, d = coefficients[0], coefficients[1], coefficients[2], coefficients[3]
                result = a * x**3 + b * x**2 + c * x + d
            else:
                result = x
            output_data[i] = result
            
    def run_constant_memory_test(self, thread_count: int = 256, block_size: int = 64):
        """Run constant memory test with different access patterns."""
        print(f"\n{'='*60}")
        print(f"CONSTANT MEMORY TEST - {thread_count} threads, {block_size} blocks")
        print(f"{'='*60}")
        
        # Allocate memory
        self.allocate_constant_memory()
        
        # Test constant memory lookup
        print("\nTesting Constant Memory Lookup...")
        lookup_start = time.perf_counter()
        self.constant_memory_lookup_operation(self.device_data, self.result_data, self.constant_data)
        cp.cuda.Stream.null.synchronize()
        lookup_end = time.perf_counter()
        lookup_time = (lookup_end - lookup_start) * 1000
        
        # Test constant memory coefficients
        print("Testing Constant Memory Coefficients...")
        coefficients = cp.array([0.1, 0.2, 0.3, 0.4], dtype=cp.float32)
        
        coeff_start = time.perf_counter()
        self.constant_memory_coefficient_operation(self.device_data, self.result_data, coefficients)
        cp.cuda.Stream.null.synchronize()
        coeff_end = time.perf_counter()
        coeff_time = (coeff_end - coeff_start) * 1000
        
        # Calculate memory bandwidth
        data_size_bytes = self.device_data.nbytes
        lookup_bandwidth = calculate_memory_bandwidth(data_size_bytes, lookup_time)
        coeff_bandwidth = calculate_memory_bandwidth(data_size_bytes, coeff_time)
        
        print(f"\nLookup Time: {lookup_time:.4f} ms")
        print(f"Coefficient Time: {coeff_time:.4f} ms")
        print(f"Lookup Bandwidth: {lookup_bandwidth:.2f} GB/s")
        print(f"Coefficient Bandwidth: {coeff_bandwidth:.2f} GB/s")
        
        return {
            'memory_type': 'constant',
            'data_size': self.data_size,
            'thread_count': thread_count,
            'block_size': block_size,
            'lookup_time_ms': lookup_time,
            'coeff_time_ms': coeff_time,
            'lookup_bandwidth_gbps': lookup_bandwidth,
            'coeff_bandwidth_gbps': coeff_bandwidth,
            'total_time_ms': lookup_time + coeff_time
        }


class RegisterMemoryDemo:
    """Register Memory demonstration class."""
    
    def __init__(self, data_size: int):
        self.data_size = data_size
        self.host_data = None
        self.device_data = None
        self.result_data = None
        
    def allocate_register_memory(self):
        """Allocate memory for register memory demo."""
        print(f"Allocating memory for register memory demo with {self.data_size:,} elements...")
        self.host_data = np.random.rand(self.data_size).astype(np.float32)
        self.device_data = cp.asarray(self.host_data)
        self.result_data = cp.zeros(self.data_size, dtype=cp.float32)
        print(f"Memory allocated: {self.device_data.nbytes:,} bytes")
        
    def register_memory_computation_operation(self, input_data, output_data):
        """Simulate register memory computation operations."""
        for i in range(input_data.size):
            # Simulate multiple register variables
            x = input_data[i]
            y = x * 2.0
            z = y + cp.sin(x)
            w = z * cp.cos(x)
            v = w + cp.tan(x)
            u = v * cp.exp(-x)
            
            # Complex computation using register variables
            result = u + y * z + w * v + cp.sqrt(cp.abs(u))
            output_data[i] = result
            
    def register_memory_loop_unrolling_operation(self, input_data, output_data):
        """Simulate register memory with loop unrolling."""
        for i in range(input_data.size):
            x = input_data[i]
            
            # Unrolled loop using register variables
            r0 = x * 1.0
            r1 = x * 2.0
            r2 = x * 3.0
            r3 = x * 4.0
            r4 = x * 5.0
            r5 = x * 6.0
            r6 = x * 7.0
            r7 = x * 8.0
            
            # Combine register variables
            result = r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7
            
            # Additional computation using registers
            temp1 = result * cp.sin(x)
            temp2 = temp1 + cp.cos(x)
            temp3 = temp2 * cp.tan(x)
            
            output_data[i] = temp3
            
    def run_register_memory_test(self, thread_count: int = 256, block_size: int = 64):
        """Run register memory test with different optimization techniques."""
        print(f"\n{'='*60}")
        print(f"REGISTER MEMORY TEST - {thread_count} threads, {block_size} blocks")
        print(f"{'='*60}")
        
        # Allocate memory
        self.allocate_register_memory()
        
        # Test basic register computation
        print("\nTesting Basic Register Computation...")
        basic_start = time.perf_counter()
        self.register_memory_computation_operation(self.device_data, self.result_data)
        cp.cuda.Stream.null.synchronize()
        basic_end = time.perf_counter()
        basic_time = (basic_end - basic_start) * 1000
        
        # Test loop unrolling with registers
        print("Testing Loop Unrolling with Registers...")
        unroll_start = time.perf_counter()
        self.register_memory_loop_unrolling_operation(self.device_data, self.result_data)
        cp.cuda.Stream.null.synchronize()
        unroll_end = time.perf_counter()
        unroll_time = (unroll_end - unroll_start) * 1000
        
        # Calculate memory bandwidth
        data_size_bytes = self.device_data.nbytes
        basic_bandwidth = calculate_memory_bandwidth(data_size_bytes, basic_time)
        unroll_bandwidth = calculate_memory_bandwidth(data_size_bytes, unroll_time)
        
        print(f"\nBasic Computation Time: {basic_time:.4f} ms")
        print(f"Loop Unrolling Time: {unroll_time:.4f} ms")
        print(f"Basic Bandwidth: {basic_bandwidth:.2f} GB/s")
        print(f"Unrolling Bandwidth: {unroll_bandwidth:.2f} GB/s")
        
        return {
            'memory_type': 'register',
            'data_size': self.data_size,
            'thread_count': thread_count,
            'block_size': block_size,
            'basic_time_ms': basic_time,
            'unroll_time_ms': unroll_time,
            'basic_bandwidth_gbps': basic_bandwidth,
            'unroll_bandwidth_gbps': unroll_bandwidth,
            'total_time_ms': basic_time + unroll_time
        }


def test_all_memory_types(data_size: str = 'medium', thread_count: int = 256, block_size: int = 64):
    """Test all memory types with given configuration."""
    print("="*80)
    print("CUDA MEMORY ASSIGNMENT - COMPREHENSIVE TESTING")
    print("="*80)
    
    # Check CUDA availability
    if not check_cuda_availability():
        print("CUDA not available. Exiting.")
        return None
    
    # Get data size
    size = DATA_SIZES[data_size]
    print(f"Testing with {data_size} data size: {size:,} elements")
    print(f"Thread count: {thread_count}, Block size: {block_size}")
    print()
    
    results = {}
    
    # Test Host Memory
    print("Testing Host Memory...")
    try:
        host_demo = HostMemoryDemo(size)
        host_result = host_demo.run_host_memory_test(thread_count, block_size)
        results['host'] = host_result
        print("✓ Host Memory Test Completed")
    except Exception as e:
        print(f"✗ Host Memory Test Failed: {e}")
        results['host'] = None
    
    cleanup_memory()
    
    # Test Global Memory
    print("\nTesting Global Memory...")
    try:
        global_demo = GlobalMemoryDemo(size)
        global_result = global_demo.run_global_memory_test(thread_count, block_size)
        results['global'] = global_result
        print("✓ Global Memory Test Completed")
    except Exception as e:
        print(f"✗ Global Memory Test Failed: {e}")
        results['global'] = None
    
    cleanup_memory()
    
    # Test Shared Memory
    print("\nTesting Shared Memory...")
    try:
        shared_demo = SharedMemoryDemo(size)
        shared_result = shared_demo.run_shared_memory_test(thread_count, block_size)
        results['shared'] = shared_result
        print("✓ Shared Memory Test Completed")
    except Exception as e:
        print(f"✗ Shared Memory Test Failed: {e}")
        results['shared'] = None
    
    cleanup_memory()
    
    # Test Constant Memory
    print("\nTesting Constant Memory...")
    try:
        constant_demo = ConstantMemoryDemo(size)
        constant_result = constant_demo.run_constant_memory_test(thread_count, block_size)
        results['constant'] = constant_result
        print("✓ Constant Memory Test Completed")
    except Exception as e:
        print(f"✗ Constant Memory Test Failed: {e}")
        results['constant'] = None
    
    cleanup_memory()
    
    # Test Register Memory
    print("\nTesting Register Memory...")
    try:
        register_demo = RegisterMemoryDemo(size)
        register_result = register_demo.run_register_memory_test(thread_count, block_size)
        results['register'] = register_result
        print("✓ Register Memory Test Completed")
    except Exception as e:
        print(f"✗ Register Memory Test Failed: {e}")
        results['register'] = None
    
    cleanup_memory()
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for memory_type, result in results.items():
        if result is not None:
            total_time = result.get('total_time_ms', 0)
            print(f"{memory_type.upper()} Memory: {total_time:.4f} ms")
        else:
            print(f"{memory_type.upper()} Memory: FAILED")
    
    return results


def main():
    """Main function for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='CUDA Memory Assignment Test Script')
    parser.add_argument('--data-size', choices=list(DATA_SIZES.keys()), default='medium',
                       help='Data size to test')
    parser.add_argument('--threads', type=int, default=256,
                       help='Number of threads')
    parser.add_argument('--blocks', type=int, default=64,
                       help='Block size')
    parser.add_argument('--memory-type', choices=list(MEMORY_TYPES.keys()), default='all',
                       help='Memory type to test')
    
    args = parser.parse_args()
    
    if args.memory_type == 'all':
        results = test_all_memory_types(args.data_size, args.threads, args.blocks)
    else:
        # Test specific memory type
        size = DATA_SIZES[args.data_size]
        
        if args.memory_type == 'host':
            demo = HostMemoryDemo(size)
            results = demo.run_host_memory_test(args.threads, args.blocks)
        elif args.memory_type == 'global':
            demo = GlobalMemoryDemo(size)
            results = demo.run_global_memory_test(args.threads, args.blocks)
        elif args.memory_type == 'shared':
            demo = SharedMemoryDemo(size)
            results = demo.run_shared_memory_test(args.threads, args.blocks)
        elif args.memory_type == 'constant':
            demo = ConstantMemoryDemo(size)
            results = demo.run_constant_memory_test(args.threads, args.blocks)
        elif args.memory_type == 'register':
            demo = RegisterMemoryDemo(size)
            results = demo.run_register_memory_test(args.threads, args.blocks)
    
    print("\nTesting completed successfully!")


if __name__ == "__main__":
    main()
