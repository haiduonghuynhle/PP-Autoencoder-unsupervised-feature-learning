#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstring>
#include <chrono>
#include <random>
#include <algorithm>
#include <iomanip>

// CUDA includes (only when compiling with nvcc)
#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUDA_KERNEL_CHECK() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA kernel error: " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#endif // __CUDACC__

// Timer utility class
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::string name;
    
public:
    Timer(const std::string& timer_name = "Timer") : name(timer_name) {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    double elapsed() const {
        auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(end_time - start_time).count();
    }
    
    void reset() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    void print(const std::string& message = "") const {
        std::cout << name << " - " << message << ": " << elapsed() << " seconds" << std::endl;
    }
};

// Random number generator
class RandomGenerator {
private:
    std::mt19937 gen;
    
public:
    RandomGenerator(unsigned int seed = 42) : gen(seed) {}
    
    // Xavier/Glorot initialization for weights
    void xavier_init(float* data, int fan_in, int fan_out, int size) {
        float std_dev = std::sqrt(2.0f / (fan_in + fan_out));
        std::normal_distribution<float> dist(0.0f, std_dev);
        for (int i = 0; i < size; ++i) {
            data[i] = dist(gen);
        }
    }
    
    // He initialization (better for ReLU)
    void he_init(float* data, int fan_in, int size) {
        float std_dev = std::sqrt(2.0f / fan_in);
        std::normal_distribution<float> dist(0.0f, std_dev);
        for (int i = 0; i < size; ++i) {
            data[i] = dist(gen);
        }
    }
    
    // Uniform random shuffle for indices
    void shuffle(std::vector<int>& indices) {
        std::shuffle(indices.begin(), indices.end(), gen);
    }
};

// Progress bar utility
inline void print_progress(int current, int total, float loss = -1.0f) {
    int bar_width = 50;
    float progress = static_cast<float>(current) / total;
    int pos = static_cast<int>(bar_width * progress);
    
    std::cout << "\r[";
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << "%";
    if (loss >= 0) {
        std::cout << " Loss: " << std::fixed << std::setprecision(6) << loss;
    }
    std::cout << std::flush;
}

// Memory utilities
inline size_t get_tensor_size(int n, int c, int h, int w) {
    return static_cast<size_t>(n) * c * h * w * sizeof(float);
}

// Constants
namespace Constants {
    // CIFAR-10 dimensions
    constexpr int CIFAR_IMG_SIZE = 32;
    constexpr int CIFAR_CHANNELS = 3;
    constexpr int CIFAR_CLASSES = 10;
    constexpr int CIFAR_TRAIN_SIZE = 50000;
    constexpr int CIFAR_TEST_SIZE = 10000;
    constexpr int CIFAR_IMG_PIXELS = CIFAR_IMG_SIZE * CIFAR_IMG_SIZE * CIFAR_CHANNELS;
    
    // Network architecture
    constexpr int LATENT_H = 8;
    constexpr int LATENT_W = 8;
    constexpr int LATENT_C = 128;
    constexpr int FEATURE_DIM = LATENT_H * LATENT_W * LATENT_C; // 8192
    
    // Training defaults
    constexpr int DEFAULT_BATCH_SIZE = 64;
    constexpr int DEFAULT_EPOCHS = 20;
    constexpr float DEFAULT_LEARNING_RATE = 0.001f;
}

#endif // UTILS_H


