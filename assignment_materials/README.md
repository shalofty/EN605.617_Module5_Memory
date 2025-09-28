# Assignment Materials

This directory contains the original assignment materials and requirements.

## Files

- **`assignment.md`** - Assignment requirements and rubric
- **`milestones.md`** - Development milestones and progress tracking
- **`MemoryAssignment.pdf`** - Official assignment PDF document

## Assignment Overview

**Course:** EN605.617 - GPU Programming  
**Assignment:** Module 5 - Memory Assignment  
**Total Points:** 100  
**Due Date:** Sunday by 11:59pm

## Requirements Summary

### Memory Types (75 points)
- **Host Memory Usage (15 pts)** - CPU-accessible memory with data transfer
- **Global Memory Usage (15 pts)** - GPU-accessible memory with access patterns
- **Shared Memory Usage (15 pts)** - Block-level shared memory with synchronization
- **Constant Memory Usage (15 pts)** - Read-only cached memory for lookup tables
- **Register Memory Usage (15 pts)** - Thread-local variables with optimization

### Additional Requirements (25 points)
- **Variable Thread Counts (5 pts)** - Multiple thread configurations tested
- **Variable Block Sizes (5 pts)** - Multiple block configurations tested
- **Command Line Interface (5 pts)** - Comprehensive CLI implemented
- **Build System (5 pts)** - Makefile and build system implemented
- **Code Quality (5 pts)** - Well-documented, organized code

## Implementation Status

Both implementations meet all requirements:

| Requirement | Points | Status |
|-------------|--------|--------|
| Host Memory Usage | 15 | Complete |
| Global Memory Usage | 15 | Complete |
| Shared Memory Usage | 15 | Complete |
| Constant Memory Usage | 15 | Complete |
| Register Memory Usage | 15 | Complete |
| Variable Thread Counts | 5 | Complete |
| Variable Block Sizes | 5 | Complete |
| Command Line Interface | 5 | Complete |
| Build System | 5 | Complete |
| Code Quality | 5 | Complete |
| **Total** | **100** | **Complete** |

## Submission Requirements

The assignment requires either:
1. **Link to commit/branch** (preferred method) - All code and artifacts included
2. **Zipped code** with images/video/links showing completion of all rubric parts

## Bonus Opportunity

One-time bonus available if code is part of final project:
- Screen capture output showing execution with different thread counts and block sizes
- Command line evidence of various configurations

## Development Milestones

See `milestones.md` for detailed development progress:
- Environment setup and CUDA configuration
- Command line interface and configuration
- Host memory implementation
- Global memory implementation
- Shared memory implementation
- Constant memory implementation
- Register memory implementation
- Variable thread and block testing
- Performance analysis and optimization
- Final integration and documentation

## Assignment PDF

The `MemoryAssignment.pdf` contains the complete official assignment document with:
- Detailed requirements and rubric
- Submission guidelines
- Grading criteria
- Technical specifications
- Examples and expectations

## Implementation Evidence

Both implementations provide comprehensive evidence of:
- All 5 memory types working correctly
- Variable thread counts (64, 128, 256, 512, 1024)
- Variable block sizes (32, 64, 128, 256, 512)
- Command line interfaces with full functionality
- Build systems with proper error handling
- High-quality, well-documented code
- Performance analysis and optimization
- Complete testing and validation
