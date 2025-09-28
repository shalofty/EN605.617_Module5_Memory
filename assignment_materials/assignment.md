# Module 5 - Memory Assignment

**Due:** Sunday by 11:59pm  
**Points:** 100  
**Submitting:** a file upload  
**Attempts:** 0 Allowed Attempts 3  
**Available:** Sep 15 at 12am - Oct 12 at 11:59pm

[MemoryAssignment.pdf](MemoryAssignment.pdf)

## Assignment Description

Create a program that utilizes all forms of CUDA memory. Your program should utilize host and global memory (arrays), register variables, constants, and shared memory. You can do one of two things:

1. Perform similar operations with the various types of CUDA device memory, or
2. Use all of the memory types in a single more complex kernel

Any comparison of timing, if the kernel code is the same, will earn extra points.

## Submission Requirements

For your assignment submission, you will need to include either:

- A link to the commit/branch for your assignment submission (preferred method), including all code and artifacts, or
- The zipped up code for the assignment and images/video/links that show your code completing all of the parts of the rubric that it is designed to complete in what is submitted for this assignment.

## Bonus Opportunity

There is one opportunity for a one-time bonus (for this and any future assessments) if the code is part of your final project. You will need to screen capture output (command line) of your code executing with different numbers of threads and block sizes.

## Rubric

| Criteria | Description | Points |
|----------|-------------|--------|
| **Host Memory Usage** | Create a program that demonstrates a large number of threads, at a minimum 64 threads with one block size, using host memory. | 15 pts |
| **Global Memory Usage** | Create a program that demonstrates a large number of threads, at a minimum 64 threads with one block size, using global memory. | 15 pts |
| **Shared Memory Usage** | Create a program that demonstrates a large number of threads, at a minimum 64 threads with one block size, using shared memory. | 15 pts |
| **Constant Memory Usage** | Create a program that demonstrates a large number of threads, at a minimum 64 threads with one block size, using constant memory. | 15 pts |
| **Register Memory Usage** | Create a program that demonstrates a large number of threads, at a minimum 64 threads with one block size, using register memory. | 15 pts |
| **Variable number of threads** | Execute or program your code to use two additional numbers of threads (capture all runs of your code). | 5 pts |
| **Variable number of Blocks** | Execute or program your code to use two additional block sizes (capture all runs of your code). | 5 pts |
| **Command Line Argument Usage** | Develop a mechanism in your code to use command line arguments to vary the number of threads and the block size. | 5 pts |
| **Use of run script and/or makefile** | Create [run.sh](Links to an external site.) or Makefile that allow for building and execution of code with various command line arguments | 5 pts |
| **Quality of Code** | Quality of code - organization of files/functions, code comments, and constants and lines no longer than 80 characters and 40 lines per function | 5 pts |

**Total Points: 100** 