# Makefile for CUDA Autoencoder Project
# CSC14120 - Parallel Programming Final Project

# Compiler settings
CXX = g++
NVCC = nvcc

# Flags
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra
NVCCFLAGS = -std=c++17 -O3 -arch=sm_75 -Xcompiler -Wall

# For Google Colab (compute capability 7.5 for T4, 8.0 for A100)
# Change -arch=sm_XX according to your GPU
# T4: sm_75, V100: sm_70, A100: sm_80, RTX 3090: sm_86

# Include directories
INCLUDES = -I./include -I./libsvm

# Library directories and libraries
LDFLAGS = -L./libsvm
LIBS = -lm

# Source files
CPU_SOURCES = src/data_loader.cpp src/layers_cpu.cpp src/autoencoder_cpu.cpp
GPU_SOURCES = src/layers_gpu.cu src/layers_gpu_optimized.cu src/autoencoder_gpu.cu
MAIN_SOURCE = src/main.cpp
SVM_SOURCE = src/svm_classifier.cpp

# Object files
CPU_OBJECTS = $(CPU_SOURCES:.cpp=.o)
GPU_OBJECTS = $(GPU_SOURCES:.cu=.o)
MAIN_OBJECT = $(MAIN_SOURCE:.cpp=.o)
SVM_OBJECT = $(SVM_SOURCE:.cpp=.o)

# Executables
CPU_TARGET = autoencoder_cpu
GPU_TARGET = autoencoder_gpu
FULL_TARGET = autoencoder_full

# Default target
all: directories $(GPU_TARGET)

# Create necessary directories
directories:
	@mkdir -p data models src include

# CPU-only build (for testing without CUDA)
cpu: directories $(CPU_TARGET)

$(CPU_TARGET): $(CPU_OBJECTS) $(SVM_OBJECT) src/main_cpu.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS)

# GPU build (main target)
$(GPU_TARGET): $(CPU_OBJECTS) $(GPU_OBJECTS) $(SVM_OBJECT) src/main_gpu.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS)

# Full build with SVM
$(FULL_TARGET): $(CPU_OBJECTS) $(GPU_OBJECTS) $(SVM_OBJECT) src/main_full.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS) -L./libsvm -lsvm

# Pattern rules
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

src/main_cpu.o: src/main.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -DCPU_ONLY -c $< -o $@

src/main_gpu.o: src/main.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

src/main_full.o: src/main.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -DWITH_SVM -c $< -o $@

# Clean
clean:
	rm -f src/*.o $(CPU_TARGET) $(GPU_TARGET) $(FULL_TARGET)

# Clean all including models
cleanall: clean
	rm -rf models/*.bin

# Download CIFAR-10 dataset
download_data:
	@echo "Downloading CIFAR-10 dataset..."
	@mkdir -p data
	@cd data && \
	if [ ! -f cifar-10-binary.tar.gz ]; then \
		curl -O https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz; \
	fi && \
	if [ ! -d cifar-10-batches-bin ]; then \
		tar -xzf cifar-10-binary.tar.gz; \
	fi
	@echo "Dataset ready in data/cifar-10-batches-bin/"

# Build LIBSVM
build_libsvm:
	@echo "Building LIBSVM..."
	@cd libsvm && make

# Run tests
test_cpu: $(CPU_TARGET)
	./$(CPU_TARGET) --test

test_gpu: $(GPU_TARGET)
	./$(GPU_TARGET) --test

# Run training
train_cpu: $(CPU_TARGET)
	./$(CPU_TARGET) --train --epochs 5 --batch-size 32

train_gpu: $(GPU_TARGET)
	./$(GPU_TARGET) --train --epochs 20 --batch-size 64

train_gpu_optimized: $(GPU_TARGET)
	./$(GPU_TARGET) --train --epochs 20 --batch-size 64 --optimized

# Run feature extraction and SVM
extract_features: $(FULL_TARGET)
	./$(FULL_TARGET) --extract-features

train_svm: $(FULL_TARGET)
	./$(FULL_TARGET) --train-svm

evaluate: $(FULL_TARGET)
	./$(FULL_TARGET) --evaluate

# Full pipeline
pipeline: download_data $(FULL_TARGET)
	./$(FULL_TARGET) --train --extract-features --train-svm --evaluate

# Help
help:
	@echo "CUDA Autoencoder - Makefile Targets"
	@echo "===================================="
	@echo ""
	@echo "Build targets:"
	@echo "  all              - Build GPU version (default)"
	@echo "  cpu              - Build CPU-only version"
	@echo "  $(FULL_TARGET)  - Build full version with SVM"
	@echo ""
	@echo "Data targets:"
	@echo "  download_data    - Download CIFAR-10 dataset"
	@echo "  build_libsvm     - Build LIBSVM library"
	@echo ""
	@echo "Run targets:"
	@echo "  train_cpu        - Train on CPU (5 epochs)"
	@echo "  train_gpu        - Train on GPU (20 epochs)"
	@echo "  train_gpu_optimized - Train on GPU with optimizations"
	@echo "  extract_features - Extract features using trained encoder"
	@echo "  train_svm        - Train SVM on extracted features"
	@echo "  evaluate         - Evaluate on test set"
	@echo "  pipeline         - Run full pipeline"
	@echo ""
	@echo "Clean targets:"
	@echo "  clean            - Remove object files and executables"
	@echo "  cleanall         - Remove everything including models"

.PHONY: all cpu clean cleanall download_data build_libsvm test_cpu test_gpu \
        train_cpu train_gpu train_gpu_optimized extract_features train_svm \
        evaluate pipeline help directories

