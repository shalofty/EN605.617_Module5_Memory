"""
CUDA Memory Assignment Command Line Interface
Module 5 - Memory Assignment CLI Implementation

This file implements the command line interface for the CUDA memory assignment,
allowing users to specify thread counts, block sizes, and memory types.
"""

import argparse
import sys
from typing import Dict, Any
from cuda_config import (
    DATA_SIZES, BLOCK_SIZES, THREAD_COUNTS, MEMORY_TYPES,
    DEFAULT_THREADS, DEFAULT_BLOCKS, DEFAULT_DATA_SIZE, 
    DEFAULT_MEMORY_TYPE, ERROR_MESSAGES
)


def parse_command_line_arguments() -> Dict[str, Any]:
    """
    Parse command line arguments for the CUDA memory assignment.
    
    Returns:
        Dictionary containing parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='CUDA Memory Assignment - Module 5',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cuda_cli.py --threads 256 --blocks 64 --memory-type global
  python cuda_cli.py --data-size large --memory-type all
  python cuda_cli.py --threads 512 --blocks 128 --memory-type shared --verbose
        """
    )
    
    # Thread configuration
    parser.add_argument(
        '--threads', '-t',
        type=int,
        default=DEFAULT_THREADS,
        help=f'Number of threads (minimum 64, default: {DEFAULT_THREADS})'
    )
    
    parser.add_argument(
        '--blocks', '-b',
        type=int,
        default=DEFAULT_BLOCKS,
        help=f'Block size (default: {DEFAULT_BLOCKS})'
    )
    
    # Data size configuration
    parser.add_argument(
        '--data-size', '-s',
        choices=list(DATA_SIZES.keys()),
        default=DEFAULT_DATA_SIZE,
        help=f'Data size preset (choices: {list(DATA_SIZES.keys())}, default: {DEFAULT_DATA_SIZE})'
    )
    
    # Memory type selection
    parser.add_argument(
        '--memory-type', '-m',
        choices=list(MEMORY_TYPES.keys()),
        default=DEFAULT_MEMORY_TYPE,
        help=f'Memory type to test (choices: {list(MEMORY_TYPES.keys())}, default: {DEFAULT_MEMORY_TYPE})'
    )
    
    # Performance and testing options
    parser.add_argument(
        '--iterations', '-i',
        type=int,
        default=100,
        help='Number of timing iterations (default: 100)'
    )
    
    parser.add_argument(
        '--test-all-configs',
        action='store_true',
        help='Test all thread and block configurations'
    )
    
    parser.add_argument(
        '--compare-performance',
        action='store_true',
        help='Compare performance across all memory types'
    )
    
    # Output options
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--output-file', '-o',
        type=str,
        help='Output file for results'
    )
    
    parser.add_argument(
        '--generate-charts',
        action='store_true',
        help='Generate performance comparison charts'
    )
    
    # Help and version
    parser.add_argument(
        '--version',
        action='version',
        version='CUDA Memory Assignment v1.0'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not validate_arguments(args):
        sys.exit(1)
    
    return vars(args)


def validate_arguments(args: argparse.Namespace) -> bool:
    """
    Validate command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        True if arguments are valid, False otherwise
    """
    # Validate thread count
    if args.threads < 64:
        print(f"Error: {ERROR_MESSAGES['invalid_thread_count']}")
        print(f"Minimum thread count is 64, got {args.threads}")
        return False
    
    # Validate block size
    if args.blocks <= 0 or args.blocks > 1024:
        print(f"Error: {ERROR_MESSAGES['invalid_block_size']}")
        print(f"Block size must be between 1 and 1024, got {args.blocks}")
        return False
    
    # Validate iterations
    if args.iterations <= 0:
        print("Error: Number of iterations must be positive")
        return False
    
    return True


def print_configuration_summary(config: Dict[str, Any]):
    """
    Print configuration summary.
    
    Args:
        config: Configuration dictionary
    """
    print("\n" + "="*60)
    print("CUDA MEMORY ASSIGNMENT CONFIGURATION")
    print("="*60)
    print(f"Thread Count: {config['threads']:,}")
    print(f"Block Size: {config['blocks']}")
    print(f"Data Size: {config['data_size']} ({DATA_SIZES[config['data_size']]:,} elements)")
    print(f"Memory Type: {config['memory_type']}")
    print(f"Timing Iterations: {config['iterations']}")
    print(f"Test All Configs: {config['test_all_configs']}")
    print(f"Compare Performance: {config['compare_performance']}")
    print(f"Verbose Output: {config['verbose']}")
    if config['output_file']:
        print(f"Output File: {config['output_file']}")
    print("="*60)


def get_test_configurations() -> list:
    """
    Get all test configurations for comprehensive testing.
    
    Returns:
        List of configuration dictionaries
    """
    configurations = []
    
    # Test different thread counts
    for thread_name, thread_count in THREAD_COUNTS.items():
        for block_name, block_size in BLOCK_SIZES.items():
            config = {
                'threads': thread_count,
                'blocks': block_size,
                'data_size': 'medium',
                'memory_type': 'all',
                'iterations': 50,  # Fewer iterations for comprehensive testing
                'test_all_configs': False,
                'compare_performance': True,
                'verbose': False,
                'output_file': None,
                'generate_charts': True
            }
            configurations.append(config)
    
    return configurations


def print_usage_examples():
    """
    Print usage examples for the command line interface.
    """
    print("\nUSAGE EXAMPLES:")
    print("-" * 40)
    print("1. Test global memory with default settings:")
    print("   python cuda_cli.py --memory-type global")
    print()
    print("2. Test all memory types with custom thread/block configuration:")
    print("   python cuda_cli.py --threads 512 --blocks 128 --memory-type all")
    print()
    print("3. Test with large data size and performance comparison:")
    print("   python cuda_cli.py --data-size large --compare-performance")
    print()
    print("4. Test all configurations with verbose output:")
    print("   python cuda_cli.py --test-all-configs --verbose")
    print()
    print("5. Generate performance charts:")
    print("   python cuda_cli.py --memory-type all --generate-charts")
    print()
    print("6. Save results to file:")
    print("   python cuda_cli.py --memory-type all --output-file results.txt")


def main():
    """
    Main function for command line interface.
    """
    try:
        config = parse_command_line_arguments()
        
        if config['verbose']:
            print_configuration_summary(config)
        
        return config
        
    except KeyboardInterrupt:
        print("\nExecution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    config = main()
    print(f"Configuration loaded: {config}")
