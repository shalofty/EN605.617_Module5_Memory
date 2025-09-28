# Makefile for CUDA Memory Assignment - Module 5
# Builds CUDA/C implementation demonstrating all 5 types of CUDA memory

# Compiler and flags
NVCC = nvcc
CC = gcc

# CUDA architecture (adjust based on your GPU)
CUDA_ARCH = -arch=sm_75

# Compiler flags
NVCC_FLAGS = $(CUDA_ARCH) -O3 -std=c++11 -Xcompiler -Wall -Xcompiler -Wextra
NVCC_FLAGS += -Xcompiler -fopenmp -Xcompiler -fPIC
NVCC_FLAGS += --ptxas-options=-v  # Show register usage
NVCC_FLAGS += --maxrregcount=255  # Maximum registers per thread

# Include directories
INCLUDES = -I/usr/local/cuda/include

# Library directories and libraries
LIB_DIRS = -L/usr/local/cuda/lib64
LIBS = -lcudart -lm

# Source files
CUDA_SRC = cuda_memory_assignment.cu
CUDA_OBJ = cuda_memory_assignment.o
TARGET = cuda_memory_assignment

# Default target
all: $(TARGET)

# Build the main executable
$(TARGET): $(CUDA_OBJ)
	@echo "Linking $(TARGET)..."
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIB_DIRS) -o $(TARGET) $(CUDA_OBJ) $(LIBS)
	@echo "Build completed successfully!"

# Compile CUDA source
$(CUDA_OBJ): $(CUDA_SRC)
	@echo "Compiling $(CUDA_SRC)..."
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $(CUDA_SRC) -o $(CUDA_OBJ)

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -f $(CUDA_OBJ) $(TARGET)
	@echo "Clean completed!"

# Install dependencies (Ubuntu/Debian)
install-deps:
	@echo "Installing CUDA dependencies..."
	sudo apt-get update
	sudo apt-get install -y nvidia-cuda-toolkit gcc g++ make
	@echo "Dependencies installed!"

# Check CUDA installation
check-cuda:
	@echo "Checking CUDA installation..."
	@which nvcc > /dev/null || (echo "Error: nvcc not found. Please install CUDA toolkit." && exit 1)
	@nvcc --version
	@nvidia-smi || echo "Warning: nvidia-smi not found. GPU may not be available."

# Run with default settings
run: $(TARGET)
	@echo "Running CUDA Memory Assignment with default settings..."
	./$(TARGET)

# Run with custom settings
run-custom: $(TARGET)
	@echo "Running CUDA Memory Assignment with custom settings..."
	./$(TARGET) --threads 512 --blocks 128 --size 1024 --verbose

# Run comprehensive testing
run-test: $(TARGET)
	@echo "Running comprehensive testing..."
	./$(TARGET) --threads 64 --blocks 32 --size 64 --verbose
	@echo ""
	./$(TARGET) --threads 128 --blocks 64 --size 256 --verbose
	@echo ""
	./$(TARGET) --threads 256 --blocks 128 --size 1024 --verbose
	@echo ""
	./$(TARGET) --threads 512 --blocks 256 --size 4096 --verbose

# Debug build
debug: NVCC_FLAGS += -g -G -DDEBUG
debug: $(TARGET)

# Release build
release: NVCC_FLAGS += -O3 -DNDEBUG
release: $(TARGET)

# Profile build
profile: NVCC_FLAGS += -pg -lineinfo
profile: $(TARGET)

# Show help
help:
	@echo "CUDA Memory Assignment - Makefile"
	@echo "=================================="
	@echo ""
	@echo "Available targets:"
	@echo "  all          - Build the main executable (default)"
	@echo "  clean        - Remove build artifacts"
	@echo "  install-deps - Install CUDA dependencies (Ubuntu/Debian)"
	@echo "  check-cuda   - Check CUDA installation"
	@echo "  run          - Run with default settings"
	@echo "  run-custom   - Run with custom settings"
	@echo "  run-test     - Run comprehensive testing"
	@echo "  debug        - Build debug version"
	@echo "  release      - Build release version"
	@echo "  profile      - Build profile version"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Usage examples:"
	@echo "  make                    # Build the executable"
	@echo "  make run                # Run with default settings"
	@echo "  make run-custom         # Run with custom settings"
	@echo "  make run-test           # Run comprehensive testing"
	@echo "  make clean              # Clean build artifacts"

# Phony targets
.PHONY: all clean install-deps check-cuda run run-custom run-test debug release profile help
