#include "autoencoder.h"
#include "data_loader.h"
#include <cuda_runtime.h>
#include <fstream>
#include <cstring>

// ============================================================================
// GPU Memory Tracking Utility
// ============================================================================

struct GPUMemoryInfo {
    size_t used_bytes;
    size_t total_bytes;
    float used_mb;
    float total_mb;
    float usage_percent;
};

GPUMemoryInfo getGPUMemoryInfo() {
    GPUMemoryInfo info;
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    info.total_bytes = total_mem;
    info.used_bytes = total_mem - free_mem;
    info.total_mb = total_mem / (1024.0f * 1024.0f);
    info.used_mb = info.used_bytes / (1024.0f * 1024.0f);
    info.usage_percent = 100.0f * info.used_bytes / total_mem;
    return info;
}

void printGPUMemory(const char* label) {
    GPUMemoryInfo info = getGPUMemoryInfo();
    printf("[GPU Memory] %s: %.2f MB / %.2f MB (%.1f%%)\n",
           label, info.used_mb, info.total_mb, info.usage_percent);
}

// Calculate theoretical memory usage for autoencoder
size_t calculateAutoencoderMemory(int batch_size) {
    size_t mem = 0;
    
    // Weights (fixed)
    mem += 256 * 3 * 3 * 3 * sizeof(float);      // enc_conv1
    mem += 256 * sizeof(float);                   // enc_conv1_bias
    mem += 128 * 256 * 3 * 3 * sizeof(float);    // enc_conv2
    mem += 128 * sizeof(float);                   // enc_conv2_bias
    mem += 128 * 128 * 3 * 3 * sizeof(float);    // dec_conv1
    mem += 128 * sizeof(float);                   // dec_conv1_bias
    mem += 256 * 128 * 3 * 3 * sizeof(float);    // dec_conv2
    mem += 256 * sizeof(float);                   // dec_conv2_bias
    mem += 3 * 256 * 3 * 3 * sizeof(float);      // dec_conv3
    mem += 3 * sizeof(float);                     // dec_conv3_bias
    
    size_t weight_mem = mem;
    
    // Gradients for weights (same size)
    mem += weight_mem;
    
    // Activations (depend on batch size)
    mem += batch_size * 3 * 32 * 32 * sizeof(float);      // input
    mem += batch_size * 256 * 32 * 32 * sizeof(float);    // enc_conv1_out
    mem += batch_size * 256 * 16 * 16 * sizeof(float);    // enc_pool1_out
    mem += batch_size * 128 * 16 * 16 * sizeof(float);    // enc_conv2_out
    mem += batch_size * 128 * 8 * 8 * sizeof(float);      // enc_pool2_out (latent)
    mem += batch_size * 128 * 8 * 8 * sizeof(float);      // dec_conv1_out
    mem += batch_size * 128 * 16 * 16 * sizeof(float);    // dec_up1_out
    mem += batch_size * 256 * 16 * 16 * sizeof(float);    // dec_conv2_out
    mem += batch_size * 256 * 32 * 32 * sizeof(float);    // dec_up2_out
    mem += batch_size * 3 * 32 * 32 * sizeof(float);      // dec_conv3_out
    mem += batch_size * 3 * 32 * 32 * sizeof(float);      // target
    
    // Pooling masks
    mem += batch_size * 256 * 16 * 16 * sizeof(int);      // pool1_mask
    mem += batch_size * 128 * 8 * 8 * sizeof(int);        // pool2_mask
    
    // Gradient activations (same as activations)
    mem += batch_size * 3 * 32 * 32 * sizeof(float);      // grad_output
    mem += batch_size * 3 * 32 * 32 * sizeof(float);      // grad_dec_conv3_out
    mem += batch_size * 256 * 32 * 32 * sizeof(float);    // grad_dec_up2_out
    mem += batch_size * 256 * 16 * 16 * sizeof(float);    // grad_dec_conv2_out
    mem += batch_size * 128 * 16 * 16 * sizeof(float);    // grad_dec_up1_out
    mem += batch_size * 128 * 8 * 8 * sizeof(float);      // grad_dec_conv1_out
    mem += batch_size * 128 * 8 * 8 * sizeof(float);      // grad_enc_pool2_out
    mem += batch_size * 128 * 16 * 16 * sizeof(float);    // grad_enc_conv2_out
    mem += batch_size * 256 * 16 * 16 * sizeof(float);    // grad_enc_pool1_out
    mem += batch_size * 256 * 32 * 32 * sizeof(float);    // grad_enc_conv1_out
    
    return mem;
}

void printMemoryBreakdown(int batch_size) {
    printf("\n============================================================\n");
    printf("GPU Memory Breakdown (batch_size=%d)\n", batch_size);
    printf("============================================================\n");
    
    size_t weight_mem = 0;
    weight_mem += 256 * 3 * 3 * 3;      // enc_conv1
    weight_mem += 256;                   // enc_conv1_bias
    weight_mem += 128 * 256 * 3 * 3;    // enc_conv2
    weight_mem += 128;                   // enc_conv2_bias
    weight_mem += 128 * 128 * 3 * 3;    // dec_conv1
    weight_mem += 128;                   // dec_conv1_bias
    weight_mem += 256 * 128 * 3 * 3;    // dec_conv2
    weight_mem += 256;                   // dec_conv2_bias
    weight_mem += 3 * 256 * 3 * 3;      // dec_conv3
    weight_mem += 3;                     // dec_conv3_bias
    weight_mem *= sizeof(float);
    
    size_t activation_mem = 0;
    activation_mem += batch_size * 3 * 32 * 32;      // input
    activation_mem += batch_size * 256 * 32 * 32;    // enc_conv1_out
    activation_mem += batch_size * 256 * 16 * 16;    // enc_pool1_out
    activation_mem += batch_size * 128 * 16 * 16;    // enc_conv2_out
    activation_mem += batch_size * 128 * 8 * 8;      // enc_pool2_out
    activation_mem += batch_size * 128 * 8 * 8;      // dec_conv1_out
    activation_mem += batch_size * 128 * 16 * 16;    // dec_up1_out
    activation_mem += batch_size * 256 * 16 * 16;    // dec_conv2_out
    activation_mem += batch_size * 256 * 32 * 32;    // dec_up2_out
    activation_mem += batch_size * 3 * 32 * 32;      // dec_conv3_out
    activation_mem += batch_size * 3 * 32 * 32;      // target
    activation_mem *= sizeof(float);
    
    size_t mask_mem = 0;
    mask_mem += batch_size * 256 * 16 * 16;
    mask_mem += batch_size * 128 * 8 * 8;
    mask_mem *= sizeof(int);
    
    size_t grad_mem = weight_mem + activation_mem;  // Gradients mirror weights + activations
    
    size_t total = weight_mem * 2 + activation_mem * 2 + mask_mem;
    
    printf("  Weights:      %8.2f MB\n", weight_mem / (1024.0 * 1024.0));
    printf("  Activations:  %8.2f MB\n", activation_mem / (1024.0 * 1024.0));
    printf("  Gradients:    %8.2f MB\n", grad_mem / (1024.0 * 1024.0));
    printf("  Pool Masks:   %8.2f MB\n", mask_mem / (1024.0 * 1024.0));
    printf("  ---------------------------------\n");
    printf("  TOTAL:        %8.2f MB\n", total / (1024.0 * 1024.0));
    printf("============================================================\n\n");
}

// ============================================================================
// GPUAutoencoder Implementation
// ============================================================================

GPUAutoencoder::GPUAutoencoder(bool optimized)
    : d_enc_conv1_weights(nullptr), d_enc_conv1_bias(nullptr),
      d_enc_conv2_weights(nullptr), d_enc_conv2_bias(nullptr),
      d_dec_conv1_weights(nullptr), d_dec_conv1_bias(nullptr),
      d_dec_conv2_weights(nullptr), d_dec_conv2_bias(nullptr),
      d_dec_conv3_weights(nullptr), d_dec_conv3_bias(nullptr),
      batch_size(0), initialized(false), use_optimized(optimized) {
    h_latent = nullptr;
    h_output = nullptr;
}

GPUAutoencoder::~GPUAutoencoder() {
    free_device_memory();
    if (h_latent) delete[] h_latent;
    if (h_output) delete[] h_output;
}

void GPUAutoencoder::allocate_device_memory() {
    // Weight sizes
    int enc_conv1_w_size = 256 * 3 * 3 * 3;
    int enc_conv2_w_size = 128 * 256 * 3 * 3;
    int dec_conv1_w_size = 128 * 128 * 3 * 3;
    int dec_conv2_w_size = 256 * 128 * 3 * 3;
    int dec_conv3_w_size = 3 * 256 * 3 * 3;
    
    // Allocate device weights
    CUDA_CHECK(cudaMalloc(&d_enc_conv1_weights, enc_conv1_w_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_enc_conv1_bias, 256 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_enc_conv2_weights, enc_conv2_w_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_enc_conv2_bias, 128 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dec_conv1_weights, dec_conv1_w_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dec_conv1_bias, 128 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dec_conv2_weights, dec_conv2_w_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dec_conv2_bias, 256 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dec_conv3_weights, dec_conv3_w_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dec_conv3_bias, 3 * sizeof(float)));
    
    // Allocate device gradients
    CUDA_CHECK(cudaMalloc(&d_grad_enc_conv1_weights, enc_conv1_w_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_enc_conv1_bias, 256 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_enc_conv2_weights, enc_conv2_w_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_enc_conv2_bias, 128 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_dec_conv1_weights, dec_conv1_w_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_dec_conv1_bias, 128 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_dec_conv2_weights, dec_conv2_w_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_dec_conv2_bias, 256 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_dec_conv3_weights, dec_conv3_w_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_dec_conv3_bias, 3 * sizeof(float)));
    
    // Allocate device activations
    CUDA_CHECK(cudaMalloc(&d_input, batch_size * 3 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_enc_conv1_out, batch_size * 256 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_enc_pool1_out, batch_size * 256 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_enc_conv2_out, batch_size * 128 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_enc_pool2_out, batch_size * 128 * 8 * 8 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dec_conv1_out, batch_size * 128 * 8 * 8 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dec_up1_out, batch_size * 128 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dec_conv2_out, batch_size * 256 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dec_up2_out, batch_size * 256 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dec_conv3_out, batch_size * 3 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_target, batch_size * 3 * 32 * 32 * sizeof(float)));
    
    // Allocate pooling masks
    CUDA_CHECK(cudaMalloc(&d_pool1_mask, batch_size * 256 * 16 * 16 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pool2_mask, batch_size * 128 * 8 * 8 * sizeof(int)));
    
    // Allocate gradient activations
    CUDA_CHECK(cudaMalloc(&d_grad_output, batch_size * 3 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_dec_conv3_out, batch_size * 3 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_dec_up2_out, batch_size * 256 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_dec_conv2_out, batch_size * 256 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_dec_up1_out, batch_size * 128 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_dec_conv1_out, batch_size * 128 * 8 * 8 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_enc_pool2_out, batch_size * 128 * 8 * 8 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_enc_conv2_out, batch_size * 128 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_enc_pool1_out, batch_size * 256 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_enc_conv1_out, batch_size * 256 * 32 * 32 * sizeof(float)));
    
    // Allocate host buffers
    h_latent = new float[batch_size * 128 * 8 * 8];
    h_output = new float[batch_size * 3 * 32 * 32];
}

void GPUAutoencoder::free_device_memory() {
    if (!initialized) return;
    
    // Free device weights
    cudaFree(d_enc_conv1_weights);
    cudaFree(d_enc_conv1_bias);
    cudaFree(d_enc_conv2_weights);
    cudaFree(d_enc_conv2_bias);
    cudaFree(d_dec_conv1_weights);
    cudaFree(d_dec_conv1_bias);
    cudaFree(d_dec_conv2_weights);
    cudaFree(d_dec_conv2_bias);
    cudaFree(d_dec_conv3_weights);
    cudaFree(d_dec_conv3_bias);
    
    // Free device gradients
    cudaFree(d_grad_enc_conv1_weights);
    cudaFree(d_grad_enc_conv1_bias);
    cudaFree(d_grad_enc_conv2_weights);
    cudaFree(d_grad_enc_conv2_bias);
    cudaFree(d_grad_dec_conv1_weights);
    cudaFree(d_grad_dec_conv1_bias);
    cudaFree(d_grad_dec_conv2_weights);
    cudaFree(d_grad_dec_conv2_bias);
    cudaFree(d_grad_dec_conv3_weights);
    cudaFree(d_grad_dec_conv3_bias);
    
    // Free device activations
    cudaFree(d_input);
    cudaFree(d_enc_conv1_out);
    cudaFree(d_enc_pool1_out);
    cudaFree(d_enc_conv2_out);
    cudaFree(d_enc_pool2_out);
    cudaFree(d_dec_conv1_out);
    cudaFree(d_dec_up1_out);
    cudaFree(d_dec_conv2_out);
    cudaFree(d_dec_up2_out);
    cudaFree(d_dec_conv3_out);
    cudaFree(d_target);
    
    // Free pooling masks
    cudaFree(d_pool1_mask);
    cudaFree(d_pool2_mask);
    
    // Free gradient activations
    cudaFree(d_grad_output);
    cudaFree(d_grad_dec_conv3_out);
    cudaFree(d_grad_dec_up2_out);
    cudaFree(d_grad_dec_conv2_out);
    cudaFree(d_grad_dec_up1_out);
    cudaFree(d_grad_dec_conv1_out);
    cudaFree(d_grad_enc_pool2_out);
    cudaFree(d_grad_enc_conv2_out);
    cudaFree(d_grad_enc_pool1_out);
    cudaFree(d_grad_enc_conv1_out);
    
    initialized = false;
}

void GPUAutoencoder::zero_gradients() {
    int enc_conv1_w_size = 256 * 3 * 3 * 3;
    int enc_conv2_w_size = 128 * 256 * 3 * 3;
    int dec_conv1_w_size = 128 * 128 * 3 * 3;
    int dec_conv2_w_size = 256 * 128 * 3 * 3;
    int dec_conv3_w_size = 3 * 256 * 3 * 3;
    
    cudaMemset(d_grad_enc_conv1_weights, 0, enc_conv1_w_size * sizeof(float));
    cudaMemset(d_grad_enc_conv1_bias, 0, 256 * sizeof(float));
    cudaMemset(d_grad_enc_conv2_weights, 0, enc_conv2_w_size * sizeof(float));
    cudaMemset(d_grad_enc_conv2_bias, 0, 128 * sizeof(float));
    cudaMemset(d_grad_dec_conv1_weights, 0, dec_conv1_w_size * sizeof(float));
    cudaMemset(d_grad_dec_conv1_bias, 0, 128 * sizeof(float));
    cudaMemset(d_grad_dec_conv2_weights, 0, dec_conv2_w_size * sizeof(float));
    cudaMemset(d_grad_dec_conv2_bias, 0, 256 * sizeof(float));
    cudaMemset(d_grad_dec_conv3_weights, 0, dec_conv3_w_size * sizeof(float));
    cudaMemset(d_grad_dec_conv3_bias, 0, 3 * sizeof(float));
}

void GPUAutoencoder::initialize(int batch_sz, unsigned int seed) {
    if (initialized) {
        free_device_memory();
    }
    
    batch_size = batch_sz;
    allocate_device_memory();
    
    // Initialize weights on host, then copy to device
    int enc_conv1_w_size = 256 * 3 * 3 * 3;
    int enc_conv2_w_size = 128 * 256 * 3 * 3;
    int dec_conv1_w_size = 128 * 128 * 3 * 3;
    int dec_conv2_w_size = 256 * 128 * 3 * 3;
    int dec_conv3_w_size = 3 * 256 * 3 * 3;
    
    float* h_weights;
    float* h_bias;
    
    // Encoder conv1
    h_weights = new float[enc_conv1_w_size];
    h_bias = new float[256];
    cpu::initialize_weights_he(h_weights, 3 * 3 * 3, enc_conv1_w_size, seed);
    cpu::initialize_bias_zero(h_bias, 256);
    CUDA_CHECK(cudaMemcpy(d_enc_conv1_weights, h_weights, enc_conv1_w_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_enc_conv1_bias, h_bias, 256 * sizeof(float), cudaMemcpyHostToDevice));
    delete[] h_weights;
    delete[] h_bias;
    
    // Encoder conv2
    h_weights = new float[enc_conv2_w_size];
    h_bias = new float[128];
    cpu::initialize_weights_he(h_weights, 256 * 3 * 3, enc_conv2_w_size, seed + 1);
    cpu::initialize_bias_zero(h_bias, 128);
    CUDA_CHECK(cudaMemcpy(d_enc_conv2_weights, h_weights, enc_conv2_w_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_enc_conv2_bias, h_bias, 128 * sizeof(float), cudaMemcpyHostToDevice));
    delete[] h_weights;
    delete[] h_bias;
    
    // Decoder conv1
    h_weights = new float[dec_conv1_w_size];
    h_bias = new float[128];
    cpu::initialize_weights_he(h_weights, 128 * 3 * 3, dec_conv1_w_size, seed + 2);
    cpu::initialize_bias_zero(h_bias, 128);
    CUDA_CHECK(cudaMemcpy(d_dec_conv1_weights, h_weights, dec_conv1_w_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dec_conv1_bias, h_bias, 128 * sizeof(float), cudaMemcpyHostToDevice));
    delete[] h_weights;
    delete[] h_bias;
    
    // Decoder conv2
    h_weights = new float[dec_conv2_w_size];
    h_bias = new float[256];
    cpu::initialize_weights_he(h_weights, 128 * 3 * 3, dec_conv2_w_size, seed + 3);
    cpu::initialize_bias_zero(h_bias, 256);
    CUDA_CHECK(cudaMemcpy(d_dec_conv2_weights, h_weights, dec_conv2_w_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dec_conv2_bias, h_bias, 256 * sizeof(float), cudaMemcpyHostToDevice));
    delete[] h_weights;
    delete[] h_bias;
    
    // Decoder conv3
    h_weights = new float[dec_conv3_w_size];
    h_bias = new float[3];
    cpu::initialize_weights_he(h_weights, 256 * 3 * 3, dec_conv3_w_size, seed + 4);
    cpu::initialize_bias_zero(h_bias, 3);
    CUDA_CHECK(cudaMemcpy(d_dec_conv3_weights, h_weights, dec_conv3_w_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dec_conv3_bias, h_bias, 3 * sizeof(float), cudaMemcpyHostToDevice));
    delete[] h_weights;
    delete[] h_bias;
    
    initialized = true;
    
    // Print memory usage info
    printMemoryBreakdown(batch_size);
    printGPUMemory("After initialization");
    
    std::cout << "GPU Autoencoder initialized with batch size " << batch_size << std::endl;
}

float GPUAutoencoder::forward(const float* h_input, float* h_output_ptr) {
    // Copy input to device (single copy - target is same as input for autoencoder)
    CUDA_CHECK(cudaMemcpy(d_input, h_input, batch_size * 3 * 32 * 32 * sizeof(float), cudaMemcpyHostToDevice));
    // Use device-to-device copy for target (much faster than host-to-device)
    CUDA_CHECK(cudaMemcpy(d_target, d_input, batch_size * 3 * 32 * 32 * sizeof(float), cudaMemcpyDeviceToDevice));
    
    if (use_optimized) {
        // Use optimized kernels
        // ==================== ENCODER ====================
        gpu_optimized::conv2d_relu_forward(d_input, d_enc_conv1_weights, d_enc_conv1_bias, d_enc_conv1_out,
                                           batch_size, 3, 32, 32, 256, 3, 1, 1);
        gpu::maxpool2d_forward(d_enc_conv1_out, d_enc_pool1_out, d_pool1_mask,
                               batch_size, 256, 32, 32, 2, 2);
        
        gpu_optimized::conv2d_relu_forward(d_enc_pool1_out, d_enc_conv2_weights, d_enc_conv2_bias, d_enc_conv2_out,
                                           batch_size, 256, 16, 16, 128, 3, 1, 1);
        gpu::maxpool2d_forward(d_enc_conv2_out, d_enc_pool2_out, d_pool2_mask,
                               batch_size, 128, 16, 16, 2, 2);
        
        // ==================== DECODER ====================
        gpu_optimized::conv2d_relu_forward(d_enc_pool2_out, d_dec_conv1_weights, d_dec_conv1_bias, d_dec_conv1_out,
                                           batch_size, 128, 8, 8, 128, 3, 1, 1);
        gpu_optimized::upsample2d_forward_coalesced(d_dec_conv1_out, d_dec_up1_out,
                                                    batch_size, 128, 8, 8, 2);
        
        gpu_optimized::conv2d_relu_forward(d_dec_up1_out, d_dec_conv2_weights, d_dec_conv2_bias, d_dec_conv2_out,
                                           batch_size, 128, 16, 16, 256, 3, 1, 1);
        gpu_optimized::upsample2d_forward_coalesced(d_dec_conv2_out, d_dec_up2_out,
                                                    batch_size, 256, 16, 16, 2);
        
        // Final conv (no ReLU)
        gpu::conv2d_forward(d_dec_up2_out, d_dec_conv3_weights, d_dec_conv3_bias, d_dec_conv3_out,
                            batch_size, 256, 32, 32, 3, 3, 1, 1);
    } else {
        // Use naive kernels
        // ==================== ENCODER ====================
        gpu::conv2d_forward(d_input, d_enc_conv1_weights, d_enc_conv1_bias, d_enc_conv1_out,
                            batch_size, 3, 32, 32, 256, 3, 1, 1);
        gpu::relu_forward(d_enc_conv1_out, d_enc_conv1_out, batch_size * 256 * 32 * 32);
        gpu::maxpool2d_forward(d_enc_conv1_out, d_enc_pool1_out, d_pool1_mask,
                               batch_size, 256, 32, 32, 2, 2);
        
        gpu::conv2d_forward(d_enc_pool1_out, d_enc_conv2_weights, d_enc_conv2_bias, d_enc_conv2_out,
                            batch_size, 256, 16, 16, 128, 3, 1, 1);
        gpu::relu_forward(d_enc_conv2_out, d_enc_conv2_out, batch_size * 128 * 16 * 16);
        gpu::maxpool2d_forward(d_enc_conv2_out, d_enc_pool2_out, d_pool2_mask,
                               batch_size, 128, 16, 16, 2, 2);
        
        // ==================== DECODER ====================
        gpu::conv2d_forward(d_enc_pool2_out, d_dec_conv1_weights, d_dec_conv1_bias, d_dec_conv1_out,
                            batch_size, 128, 8, 8, 128, 3, 1, 1);
        gpu::relu_forward(d_dec_conv1_out, d_dec_conv1_out, batch_size * 128 * 8 * 8);
        gpu::upsample2d_forward(d_dec_conv1_out, d_dec_up1_out,
                                batch_size, 128, 8, 8, 2);
        
        gpu::conv2d_forward(d_dec_up1_out, d_dec_conv2_weights, d_dec_conv2_bias, d_dec_conv2_out,
                            batch_size, 128, 16, 16, 256, 3, 1, 1);
        gpu::relu_forward(d_dec_conv2_out, d_dec_conv2_out, batch_size * 256 * 16 * 16);
        gpu::upsample2d_forward(d_dec_conv2_out, d_dec_up2_out,
                                batch_size, 256, 16, 16, 2);
        
        // Final conv (no ReLU)
        gpu::conv2d_forward(d_dec_up2_out, d_dec_conv3_weights, d_dec_conv3_bias, d_dec_conv3_out,
                            batch_size, 256, 32, 32, 3, 3, 1, 1);
    }
    
    // Copy output to host if requested
    if (h_output_ptr != nullptr) {
        CUDA_CHECK(cudaMemcpy(h_output_ptr, d_dec_conv3_out, batch_size * 3 * 32 * 32 * sizeof(float), cudaMemcpyDeviceToHost));
    }
    
    // Compute loss
    return gpu::mse_loss_forward(d_dec_conv3_out, d_target, batch_size * 3 * 32 * 32);
}

void GPUAutoencoder::encode(const float* h_input, float* h_features) {
    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, batch_size * 3 * 32 * 32 * sizeof(float), cudaMemcpyHostToDevice));
    
    // Run encoder only
    if (use_optimized) {
        gpu_optimized::conv2d_relu_forward(d_input, d_enc_conv1_weights, d_enc_conv1_bias, d_enc_conv1_out,
                                           batch_size, 3, 32, 32, 256, 3, 1, 1);
        gpu::maxpool2d_forward(d_enc_conv1_out, d_enc_pool1_out, d_pool1_mask,
                               batch_size, 256, 32, 32, 2, 2);
        
        gpu_optimized::conv2d_relu_forward(d_enc_pool1_out, d_enc_conv2_weights, d_enc_conv2_bias, d_enc_conv2_out,
                                           batch_size, 256, 16, 16, 128, 3, 1, 1);
        gpu::maxpool2d_forward(d_enc_conv2_out, d_enc_pool2_out, d_pool2_mask,
                               batch_size, 128, 16, 16, 2, 2);
    } else {
        gpu::conv2d_forward(d_input, d_enc_conv1_weights, d_enc_conv1_bias, d_enc_conv1_out,
                            batch_size, 3, 32, 32, 256, 3, 1, 1);
        gpu::relu_forward(d_enc_conv1_out, d_enc_conv1_out, batch_size * 256 * 32 * 32);
        gpu::maxpool2d_forward(d_enc_conv1_out, d_enc_pool1_out, d_pool1_mask,
                               batch_size, 256, 32, 32, 2, 2);
        
        gpu::conv2d_forward(d_enc_pool1_out, d_enc_conv2_weights, d_enc_conv2_bias, d_enc_conv2_out,
                            batch_size, 256, 16, 16, 128, 3, 1, 1);
        gpu::relu_forward(d_enc_conv2_out, d_enc_conv2_out, batch_size * 128 * 16 * 16);
        gpu::maxpool2d_forward(d_enc_conv2_out, d_enc_pool2_out, d_pool2_mask,
                               batch_size, 128, 16, 16, 2, 2);
    }
    
    // Copy features to host
    CUDA_CHECK(cudaMemcpy(h_features, d_enc_pool2_out, batch_size * Constants::FEATURE_DIM * sizeof(float), cudaMemcpyDeviceToHost));
}

void GPUAutoencoder::backward(const float* h_target) {
    zero_gradients();
    
    // MSE loss backward
    gpu::mse_loss_backward(d_dec_conv3_out, d_target, d_grad_dec_conv3_out, batch_size * 3 * 32 * 32);
    
    if (use_optimized) {
        // ==================== DECODER BACKWARD (OPTIMIZED) ====================
        
        // Conv5 backward
        gpu_optimized::conv2d_backward_tiled(d_dec_up2_out, d_dec_conv3_weights, d_grad_dec_conv3_out,
                             d_grad_dec_up2_out, d_grad_dec_conv3_weights, d_grad_dec_conv3_bias,
                             batch_size, 256, 32, 32, 3, 3, 1, 1);
        
        // Upsample2 backward
        gpu::upsample2d_backward(d_grad_dec_up2_out, d_grad_dec_conv2_out,
                                 batch_size, 256, 16, 16, 2);
        
        // ReLU4 backward
        gpu_optimized::relu_backward_vectorized(d_dec_conv2_out, d_grad_dec_conv2_out, d_grad_dec_conv2_out,
                           batch_size * 256 * 16 * 16);
        
        // Conv4 backward
        gpu_optimized::conv2d_backward_tiled(d_dec_up1_out, d_dec_conv2_weights, d_grad_dec_conv2_out,
                             d_grad_dec_up1_out, d_grad_dec_conv2_weights, d_grad_dec_conv2_bias,
                             batch_size, 128, 16, 16, 256, 3, 1, 1);
        
        // Upsample1 backward
        gpu::upsample2d_backward(d_grad_dec_up1_out, d_grad_dec_conv1_out,
                                 batch_size, 128, 8, 8, 2);
        
        // ReLU3 backward
        gpu_optimized::relu_backward_vectorized(d_dec_conv1_out, d_grad_dec_conv1_out, d_grad_dec_conv1_out,
                           batch_size * 128 * 8 * 8);
        
        // Conv3 backward
        gpu_optimized::conv2d_backward_tiled(d_enc_pool2_out, d_dec_conv1_weights, d_grad_dec_conv1_out,
                             d_grad_enc_pool2_out, d_grad_dec_conv1_weights, d_grad_dec_conv1_bias,
                             batch_size, 128, 8, 8, 128, 3, 1, 1);
        
        // ==================== ENCODER BACKWARD (OPTIMIZED) ====================
        
        // MaxPool2 backward
        gpu::maxpool2d_backward(d_grad_enc_pool2_out, d_pool2_mask, d_grad_enc_conv2_out,
                                batch_size, 128, 16, 16, 2, 2);
        
        // ReLU2 backward
        gpu_optimized::relu_backward_vectorized(d_enc_conv2_out, d_grad_enc_conv2_out, d_grad_enc_conv2_out,
                           batch_size * 128 * 16 * 16);
        
        // Conv2 backward
        gpu_optimized::conv2d_backward_tiled(d_enc_pool1_out, d_enc_conv2_weights, d_grad_enc_conv2_out,
                             d_grad_enc_pool1_out, d_grad_enc_conv2_weights, d_grad_enc_conv2_bias,
                             batch_size, 256, 16, 16, 128, 3, 1, 1);
        
        // MaxPool1 backward
        gpu::maxpool2d_backward(d_grad_enc_pool1_out, d_pool1_mask, d_grad_enc_conv1_out,
                                batch_size, 256, 32, 32, 2, 2);
        
        // ReLU1 backward
        gpu_optimized::relu_backward_vectorized(d_enc_conv1_out, d_grad_enc_conv1_out, d_grad_enc_conv1_out,
                           batch_size * 256 * 32 * 32);
        
        // Conv1 backward
        float* d_dummy_grad_input;
        CUDA_CHECK(cudaMalloc(&d_dummy_grad_input, batch_size * 3 * 32 * 32 * sizeof(float)));
        gpu_optimized::conv2d_backward_tiled(d_input, d_enc_conv1_weights, d_grad_enc_conv1_out,
                             d_dummy_grad_input, d_grad_enc_conv1_weights, d_grad_enc_conv1_bias,
                             batch_size, 3, 32, 32, 256, 3, 1, 1);
        cudaFree(d_dummy_grad_input);
    } else {
        // ==================== DECODER BACKWARD (NAIVE) ====================
        
        // Conv5 backward
        gpu::conv2d_backward(d_dec_up2_out, d_dec_conv3_weights, d_grad_dec_conv3_out,
                             d_grad_dec_up2_out, d_grad_dec_conv3_weights, d_grad_dec_conv3_bias,
                             batch_size, 256, 32, 32, 3, 3, 1, 1);
        
        // Upsample2 backward
        gpu::upsample2d_backward(d_grad_dec_up2_out, d_grad_dec_conv2_out,
                                 batch_size, 256, 16, 16, 2);
        
        // ReLU4 backward
        gpu::relu_backward(d_dec_conv2_out, d_grad_dec_conv2_out, d_grad_dec_conv2_out,
                           batch_size * 256 * 16 * 16);
        
        // Conv4 backward
        gpu::conv2d_backward(d_dec_up1_out, d_dec_conv2_weights, d_grad_dec_conv2_out,
                             d_grad_dec_up1_out, d_grad_dec_conv2_weights, d_grad_dec_conv2_bias,
                             batch_size, 128, 16, 16, 256, 3, 1, 1);
        
        // Upsample1 backward
        gpu::upsample2d_backward(d_grad_dec_up1_out, d_grad_dec_conv1_out,
                                 batch_size, 128, 8, 8, 2);
        
        // ReLU3 backward
        gpu::relu_backward(d_dec_conv1_out, d_grad_dec_conv1_out, d_grad_dec_conv1_out,
                           batch_size * 128 * 8 * 8);
        
        // Conv3 backward
        gpu::conv2d_backward(d_enc_pool2_out, d_dec_conv1_weights, d_grad_dec_conv1_out,
                             d_grad_enc_pool2_out, d_grad_dec_conv1_weights, d_grad_dec_conv1_bias,
                             batch_size, 128, 8, 8, 128, 3, 1, 1);
        
        // ==================== ENCODER BACKWARD (NAIVE) ====================
        
        // MaxPool2 backward
        gpu::maxpool2d_backward(d_grad_enc_pool2_out, d_pool2_mask, d_grad_enc_conv2_out,
                                batch_size, 128, 16, 16, 2, 2);
        
        // ReLU2 backward
        gpu::relu_backward(d_enc_conv2_out, d_grad_enc_conv2_out, d_grad_enc_conv2_out,
                           batch_size * 128 * 16 * 16);
        
        // Conv2 backward
        gpu::conv2d_backward(d_enc_pool1_out, d_enc_conv2_weights, d_grad_enc_conv2_out,
                             d_grad_enc_pool1_out, d_grad_enc_conv2_weights, d_grad_enc_conv2_bias,
                             batch_size, 256, 16, 16, 128, 3, 1, 1);
        
        // MaxPool1 backward
        gpu::maxpool2d_backward(d_grad_enc_pool1_out, d_pool1_mask, d_grad_enc_conv1_out,
                                batch_size, 256, 32, 32, 2, 2);
        
        // ReLU1 backward
        gpu::relu_backward(d_enc_conv1_out, d_grad_enc_conv1_out, d_grad_enc_conv1_out,
                           batch_size * 256 * 32 * 32);
        
        // Conv1 backward (we need grad_input for completeness, but don't use it)
        float* d_dummy_grad_input;
        CUDA_CHECK(cudaMalloc(&d_dummy_grad_input, batch_size * 3 * 32 * 32 * sizeof(float)));
        gpu::conv2d_backward(d_input, d_enc_conv1_weights, d_grad_enc_conv1_out,
                             d_dummy_grad_input, d_grad_enc_conv1_weights, d_grad_enc_conv1_bias,
                             batch_size, 3, 32, 32, 256, 3, 1, 1);
        cudaFree(d_dummy_grad_input);
    }
}

void GPUAutoencoder::update_weights(float learning_rate) {
    if (use_optimized) {
        gpu_optimized::sgd_update_vectorized(d_enc_conv1_weights, d_grad_enc_conv1_weights, learning_rate, 256 * 3 * 3 * 3);
        gpu_optimized::sgd_update_vectorized(d_enc_conv1_bias, d_grad_enc_conv1_bias, learning_rate, 256);
        gpu_optimized::sgd_update_vectorized(d_enc_conv2_weights, d_grad_enc_conv2_weights, learning_rate, 128 * 256 * 3 * 3);
        gpu_optimized::sgd_update_vectorized(d_enc_conv2_bias, d_grad_enc_conv2_bias, learning_rate, 128);
        gpu_optimized::sgd_update_vectorized(d_dec_conv1_weights, d_grad_dec_conv1_weights, learning_rate, 128 * 128 * 3 * 3);
        gpu_optimized::sgd_update_vectorized(d_dec_conv1_bias, d_grad_dec_conv1_bias, learning_rate, 128);
        gpu_optimized::sgd_update_vectorized(d_dec_conv2_weights, d_grad_dec_conv2_weights, learning_rate, 256 * 128 * 3 * 3);
        gpu_optimized::sgd_update_vectorized(d_dec_conv2_bias, d_grad_dec_conv2_bias, learning_rate, 256);
        gpu_optimized::sgd_update_vectorized(d_dec_conv3_weights, d_grad_dec_conv3_weights, learning_rate, 3 * 256 * 3 * 3);
        gpu_optimized::sgd_update_vectorized(d_dec_conv3_bias, d_grad_dec_conv3_bias, learning_rate, 3);
    } else {
        gpu::sgd_update(d_enc_conv1_weights, d_grad_enc_conv1_weights, learning_rate, 256 * 3 * 3 * 3);
        gpu::sgd_update(d_enc_conv1_bias, d_grad_enc_conv1_bias, learning_rate, 256);
        gpu::sgd_update(d_enc_conv2_weights, d_grad_enc_conv2_weights, learning_rate, 128 * 256 * 3 * 3);
        gpu::sgd_update(d_enc_conv2_bias, d_grad_enc_conv2_bias, learning_rate, 128);
        gpu::sgd_update(d_dec_conv1_weights, d_grad_dec_conv1_weights, learning_rate, 128 * 128 * 3 * 3);
        gpu::sgd_update(d_dec_conv1_bias, d_grad_dec_conv1_bias, learning_rate, 128);
        gpu::sgd_update(d_dec_conv2_weights, d_grad_dec_conv2_weights, learning_rate, 256 * 128 * 3 * 3);
        gpu::sgd_update(d_dec_conv2_bias, d_grad_dec_conv2_bias, learning_rate, 256);
        gpu::sgd_update(d_dec_conv3_weights, d_grad_dec_conv3_weights, learning_rate, 3 * 256 * 3 * 3);
        gpu::sgd_update(d_dec_conv3_bias, d_grad_dec_conv3_bias, learning_rate, 3);
    }
}

void GPUAutoencoder::get_latent(float* h_features) const {
    CUDA_CHECK(cudaMemcpy(h_features, d_enc_pool2_out, batch_size * Constants::FEATURE_DIM * sizeof(float), cudaMemcpyDeviceToHost));
}

// ============================================================================
// Stream-aware methods for opt-v2
// ============================================================================

void GPUAutoencoder::copy_input_async(const float* h_input, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(d_input, h_input, batch_size * 3 * 32 * 32 * sizeof(float), 
                                cudaMemcpyHostToDevice, stream));
    // Copy target from input (autoencoder reconstructs input)
    CUDA_CHECK(cudaMemcpyAsync(d_target, d_input, batch_size * 3 * 32 * 32 * sizeof(float), 
                                cudaMemcpyDeviceToDevice, stream));
}

float GPUAutoencoder::forward_async(cudaStream_t stream) {
    // Note: All kernels are launched on the default stream, which is synchronized
    // For full async, we would need stream-aware kernel wrappers
    // This implementation synchronizes the stream before forward to ensure input is ready
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // ==================== ENCODER ====================
    gpu_optimized::conv2d_relu_forward(d_input, d_enc_conv1_weights, d_enc_conv1_bias, d_enc_conv1_out,
                                       batch_size, 3, 32, 32, 256, 3, 1, 1);
    gpu::maxpool2d_forward(d_enc_conv1_out, d_enc_pool1_out, d_pool1_mask,
                           batch_size, 256, 32, 32, 2, 2);
    
    gpu_optimized::conv2d_relu_forward(d_enc_pool1_out, d_enc_conv2_weights, d_enc_conv2_bias, d_enc_conv2_out,
                                       batch_size, 256, 16, 16, 128, 3, 1, 1);
    gpu::maxpool2d_forward(d_enc_conv2_out, d_enc_pool2_out, d_pool2_mask,
                           batch_size, 128, 16, 16, 2, 2);
    
    // ==================== DECODER ====================
    gpu_optimized::conv2d_relu_forward(d_enc_pool2_out, d_dec_conv1_weights, d_dec_conv1_bias, d_dec_conv1_out,
                                       batch_size, 128, 8, 8, 128, 3, 1, 1);
    gpu_optimized::upsample2d_forward_coalesced(d_dec_conv1_out, d_dec_up1_out,
                                                batch_size, 128, 8, 8, 2);
    
    gpu_optimized::conv2d_relu_forward(d_dec_up1_out, d_dec_conv2_weights, d_dec_conv2_bias, d_dec_conv2_out,
                                       batch_size, 128, 16, 16, 256, 3, 1, 1);
    gpu_optimized::upsample2d_forward_coalesced(d_dec_conv2_out, d_dec_up2_out,
                                                batch_size, 256, 16, 16, 2);
    
    // Final conv (no ReLU)
    gpu::conv2d_forward(d_dec_up2_out, d_dec_conv3_weights, d_dec_conv3_bias, d_dec_conv3_out,
                        batch_size, 256, 32, 32, 3, 3, 1, 1);
    
    // Compute MSE loss
    float loss = gpu::mse_loss_forward(d_dec_conv3_out, d_target, batch_size * 3 * 32 * 32);
    // Compute gradient for backward pass
    gpu::mse_loss_backward(d_dec_conv3_out, d_target, d_grad_output, batch_size * 3 * 32 * 32);
    return loss;
}

void GPUAutoencoder::backward_async(cudaStream_t stream) {
    // Backward pass uses optimized kernels
    // Dec conv3 backward
    gpu_optimized::conv2d_backward_tiled(d_dec_up2_out, d_dec_conv3_weights, d_grad_output,
                                         d_grad_dec_up2_out, d_grad_dec_conv3_weights, d_grad_dec_conv3_bias,
                                         batch_size, 256, 32, 32, 3, 3, 1, 1);
    
    // Upsample2 backward
    gpu::upsample2d_backward(d_grad_dec_up2_out, d_grad_dec_conv2_out,
                             batch_size, 256, 16, 16, 2);
    
    // Dec conv2 backward (through ReLU)
    gpu_optimized::relu_backward_vectorized(d_dec_conv2_out, d_grad_dec_conv2_out, d_grad_dec_conv2_out,
                                            batch_size * 256 * 16 * 16);
    
    gpu_optimized::conv2d_backward_tiled(d_dec_up1_out, d_dec_conv2_weights, d_grad_dec_conv2_out,
                                         d_grad_dec_up1_out, d_grad_dec_conv2_weights, d_grad_dec_conv2_bias,
                                         batch_size, 128, 16, 16, 256, 3, 1, 1);
    
    // Upsample1 backward
    gpu::upsample2d_backward(d_grad_dec_up1_out, d_grad_dec_conv1_out,
                             batch_size, 128, 8, 8, 2);
    
    // Dec conv1 backward (through ReLU)
    gpu_optimized::relu_backward_vectorized(d_dec_conv1_out, d_grad_dec_conv1_out, d_grad_dec_conv1_out,
                                            batch_size * 128 * 8 * 8);
    
    gpu_optimized::conv2d_backward_tiled(d_enc_pool2_out, d_dec_conv1_weights, d_grad_dec_conv1_out,
                                         d_grad_enc_pool2_out, d_grad_dec_conv1_weights, d_grad_dec_conv1_bias,
                                         batch_size, 128, 8, 8, 128, 3, 1, 1);
    
    // Pool2 backward
    gpu::maxpool2d_backward(d_grad_enc_pool2_out, d_pool2_mask, d_grad_enc_conv2_out,
                            batch_size, 128, 16, 16, 2, 2);
    
    // Enc conv2 backward (through ReLU)
    gpu_optimized::relu_backward_vectorized(d_enc_conv2_out, d_grad_enc_conv2_out, d_grad_enc_conv2_out,
                                            batch_size * 128 * 16 * 16);
    
    gpu_optimized::conv2d_backward_tiled(d_enc_pool1_out, d_enc_conv2_weights, d_grad_enc_conv2_out,
                                         d_grad_enc_pool1_out, d_grad_enc_conv2_weights, d_grad_enc_conv2_bias,
                                         batch_size, 256, 16, 16, 128, 3, 1, 1);
    
    // Pool1 backward
    gpu::maxpool2d_backward(d_grad_enc_pool1_out, d_pool1_mask, d_grad_enc_conv1_out,
                            batch_size, 256, 32, 32, 2, 2);
    
    // Enc conv1 backward (through ReLU)
    gpu_optimized::relu_backward_vectorized(d_enc_conv1_out, d_grad_enc_conv1_out, d_grad_enc_conv1_out,
                                            batch_size * 256 * 32 * 32);
    
    // Skip input gradient (not needed for training)
}

void GPUAutoencoder::update_weights_async(float learning_rate, cudaStream_t stream) {
    gpu_optimized::sgd_update_vectorized(d_enc_conv1_weights, d_grad_enc_conv1_weights, learning_rate, 256 * 3 * 3 * 3);
    gpu_optimized::sgd_update_vectorized(d_enc_conv1_bias, d_grad_enc_conv1_bias, learning_rate, 256);
    gpu_optimized::sgd_update_vectorized(d_enc_conv2_weights, d_grad_enc_conv2_weights, learning_rate, 128 * 256 * 3 * 3);
    gpu_optimized::sgd_update_vectorized(d_enc_conv2_bias, d_grad_enc_conv2_bias, learning_rate, 128);
    gpu_optimized::sgd_update_vectorized(d_dec_conv1_weights, d_grad_dec_conv1_weights, learning_rate, 128 * 128 * 3 * 3);
    gpu_optimized::sgd_update_vectorized(d_dec_conv1_bias, d_grad_dec_conv1_bias, learning_rate, 128);
    gpu_optimized::sgd_update_vectorized(d_dec_conv2_weights, d_grad_dec_conv2_weights, learning_rate, 256 * 128 * 3 * 3);
    gpu_optimized::sgd_update_vectorized(d_dec_conv2_bias, d_grad_dec_conv2_bias, learning_rate, 256);
    gpu_optimized::sgd_update_vectorized(d_dec_conv3_weights, d_grad_dec_conv3_weights, learning_rate, 3 * 256 * 3 * 3);
    gpu_optimized::sgd_update_vectorized(d_dec_conv3_bias, d_grad_dec_conv3_bias, learning_rate, 3);
}

bool GPUAutoencoder::save_weights(const std::string& filepath) const {
    // Copy weights from device to host and save
    int enc_conv1_w_size = 256 * 3 * 3 * 3;
    int enc_conv2_w_size = 128 * 256 * 3 * 3;
    int dec_conv1_w_size = 128 * 128 * 3 * 3;
    int dec_conv2_w_size = 256 * 128 * 3 * 3;
    int dec_conv3_w_size = 3 * 256 * 3 * 3;
    
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file for writing: " << filepath << std::endl;
        return false;
    }
    
    int magic = 0xAE2024;
    int version = 1;
    file.write(reinterpret_cast<const char*>(&magic), sizeof(int));
    file.write(reinterpret_cast<const char*>(&version), sizeof(int));
    
    float* h_buffer;
    
    // Save encoder conv1
    h_buffer = new float[enc_conv1_w_size];
    cudaMemcpy(h_buffer, d_enc_conv1_weights, enc_conv1_w_size * sizeof(float), cudaMemcpyDeviceToHost);
    file.write(reinterpret_cast<char*>(h_buffer), enc_conv1_w_size * sizeof(float));
    delete[] h_buffer;
    
    h_buffer = new float[256];
    cudaMemcpy(h_buffer, d_enc_conv1_bias, 256 * sizeof(float), cudaMemcpyDeviceToHost);
    file.write(reinterpret_cast<char*>(h_buffer), 256 * sizeof(float));
    delete[] h_buffer;
    
    // Save encoder conv2
    h_buffer = new float[enc_conv2_w_size];
    cudaMemcpy(h_buffer, d_enc_conv2_weights, enc_conv2_w_size * sizeof(float), cudaMemcpyDeviceToHost);
    file.write(reinterpret_cast<char*>(h_buffer), enc_conv2_w_size * sizeof(float));
    delete[] h_buffer;
    
    h_buffer = new float[128];
    cudaMemcpy(h_buffer, d_enc_conv2_bias, 128 * sizeof(float), cudaMemcpyDeviceToHost);
    file.write(reinterpret_cast<char*>(h_buffer), 128 * sizeof(float));
    delete[] h_buffer;
    
    // Save decoder conv1
    h_buffer = new float[dec_conv1_w_size];
    cudaMemcpy(h_buffer, d_dec_conv1_weights, dec_conv1_w_size * sizeof(float), cudaMemcpyDeviceToHost);
    file.write(reinterpret_cast<char*>(h_buffer), dec_conv1_w_size * sizeof(float));
    delete[] h_buffer;
    
    h_buffer = new float[128];
    cudaMemcpy(h_buffer, d_dec_conv1_bias, 128 * sizeof(float), cudaMemcpyDeviceToHost);
    file.write(reinterpret_cast<char*>(h_buffer), 128 * sizeof(float));
    delete[] h_buffer;
    
    // Save decoder conv2
    h_buffer = new float[dec_conv2_w_size];
    cudaMemcpy(h_buffer, d_dec_conv2_weights, dec_conv2_w_size * sizeof(float), cudaMemcpyDeviceToHost);
    file.write(reinterpret_cast<char*>(h_buffer), dec_conv2_w_size * sizeof(float));
    delete[] h_buffer;
    
    h_buffer = new float[256];
    cudaMemcpy(h_buffer, d_dec_conv2_bias, 256 * sizeof(float), cudaMemcpyDeviceToHost);
    file.write(reinterpret_cast<char*>(h_buffer), 256 * sizeof(float));
    delete[] h_buffer;
    
    // Save decoder conv3
    h_buffer = new float[dec_conv3_w_size];
    cudaMemcpy(h_buffer, d_dec_conv3_weights, dec_conv3_w_size * sizeof(float), cudaMemcpyDeviceToHost);
    file.write(reinterpret_cast<char*>(h_buffer), dec_conv3_w_size * sizeof(float));
    delete[] h_buffer;
    
    h_buffer = new float[3];
    cudaMemcpy(h_buffer, d_dec_conv3_bias, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    file.write(reinterpret_cast<char*>(h_buffer), 3 * sizeof(float));
    delete[] h_buffer;
    
    file.close();
    std::cout << "Weights saved to " << filepath << std::endl;
    return true;
}

bool GPUAutoencoder::load_weights(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file for reading: " << filepath << std::endl;
        return false;
    }
    
    int magic, version;
    file.read(reinterpret_cast<char*>(&magic), sizeof(int));
    file.read(reinterpret_cast<char*>(&version), sizeof(int));
    
    if (magic != 0xAE2024) {
        std::cerr << "Error: Invalid model file format" << std::endl;
        return false;
    }
    
    int enc_conv1_w_size = 256 * 3 * 3 * 3;
    int enc_conv2_w_size = 128 * 256 * 3 * 3;
    int dec_conv1_w_size = 128 * 128 * 3 * 3;
    int dec_conv2_w_size = 256 * 128 * 3 * 3;
    int dec_conv3_w_size = 3 * 256 * 3 * 3;
    
    float* h_buffer;
    
    // Load encoder conv1
    h_buffer = new float[enc_conv1_w_size];
    file.read(reinterpret_cast<char*>(h_buffer), enc_conv1_w_size * sizeof(float));
    cudaMemcpy(d_enc_conv1_weights, h_buffer, enc_conv1_w_size * sizeof(float), cudaMemcpyHostToDevice);
    delete[] h_buffer;
    
    h_buffer = new float[256];
    file.read(reinterpret_cast<char*>(h_buffer), 256 * sizeof(float));
    cudaMemcpy(d_enc_conv1_bias, h_buffer, 256 * sizeof(float), cudaMemcpyHostToDevice);
    delete[] h_buffer;
    
    // Load encoder conv2
    h_buffer = new float[enc_conv2_w_size];
    file.read(reinterpret_cast<char*>(h_buffer), enc_conv2_w_size * sizeof(float));
    cudaMemcpy(d_enc_conv2_weights, h_buffer, enc_conv2_w_size * sizeof(float), cudaMemcpyHostToDevice);
    delete[] h_buffer;
    
    h_buffer = new float[128];
    file.read(reinterpret_cast<char*>(h_buffer), 128 * sizeof(float));
    cudaMemcpy(d_enc_conv2_bias, h_buffer, 128 * sizeof(float), cudaMemcpyHostToDevice);
    delete[] h_buffer;
    
    // Load decoder conv1
    h_buffer = new float[dec_conv1_w_size];
    file.read(reinterpret_cast<char*>(h_buffer), dec_conv1_w_size * sizeof(float));
    cudaMemcpy(d_dec_conv1_weights, h_buffer, dec_conv1_w_size * sizeof(float), cudaMemcpyHostToDevice);
    delete[] h_buffer;
    
    h_buffer = new float[128];
    file.read(reinterpret_cast<char*>(h_buffer), 128 * sizeof(float));
    cudaMemcpy(d_dec_conv1_bias, h_buffer, 128 * sizeof(float), cudaMemcpyHostToDevice);
    delete[] h_buffer;
    
    // Load decoder conv2
    h_buffer = new float[dec_conv2_w_size];
    file.read(reinterpret_cast<char*>(h_buffer), dec_conv2_w_size * sizeof(float));
    cudaMemcpy(d_dec_conv2_weights, h_buffer, dec_conv2_w_size * sizeof(float), cudaMemcpyHostToDevice);
    delete[] h_buffer;
    
    h_buffer = new float[256];
    file.read(reinterpret_cast<char*>(h_buffer), 256 * sizeof(float));
    cudaMemcpy(d_dec_conv2_bias, h_buffer, 256 * sizeof(float), cudaMemcpyHostToDevice);
    delete[] h_buffer;
    
    // Load decoder conv3
    h_buffer = new float[dec_conv3_w_size];
    file.read(reinterpret_cast<char*>(h_buffer), dec_conv3_w_size * sizeof(float));
    cudaMemcpy(d_dec_conv3_weights, h_buffer, dec_conv3_w_size * sizeof(float), cudaMemcpyHostToDevice);
    delete[] h_buffer;
    
    h_buffer = new float[3];
    file.read(reinterpret_cast<char*>(h_buffer), 3 * sizeof(float));
    cudaMemcpy(d_dec_conv3_bias, h_buffer, 3 * sizeof(float), cudaMemcpyHostToDevice);
    delete[] h_buffer;
    
    file.close();
    std::cout << "Weights loaded from " << filepath << std::endl;
    return true;
}

// ============================================================================
// GPU Training Functions
// ============================================================================

void train_gpu(CIFAR10Dataset& dataset, const TrainingConfig& config, TrainingStats& stats) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "GPU Training - BASIC (Naive Parallelization)" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Phase: GPU Basic" << std::endl;
    std::cout << "Optimization: Naive CUDA parallelization" << std::endl;
    std::cout << "Speedup target: ~10x vs CPU" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Batch size: " << config.batch_size << std::endl;
    std::cout << "Epochs: " << config.epochs << std::endl;
    std::cout << "Learning rate: " << config.learning_rate << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // Initialize autoencoder
    GPUAutoencoder autoencoder(false);
    autoencoder.initialize(config.batch_size, config.seed);
    
    // Calculate and display GPU memory usage
    size_t weights_mem = 0;
    weights_mem += 256 * 3 * 3 * 3 * sizeof(float);    // enc_conv1_weights
    weights_mem += 256 * sizeof(float);                 // enc_conv1_bias
    weights_mem += 128 * 256 * 3 * 3 * sizeof(float);  // enc_conv2_weights
    weights_mem += 128 * sizeof(float);                 // enc_conv2_bias
    weights_mem += 128 * 128 * 3 * 3 * sizeof(float);  // dec_conv1_weights
    weights_mem += 128 * sizeof(float);                 // dec_conv1_bias
    weights_mem += 256 * 128 * 3 * 3 * sizeof(float);  // dec_conv2_weights
    weights_mem += 256 * sizeof(float);                 // dec_conv2_bias
    weights_mem += 3 * 256 * 3 * 3 * sizeof(float);    // dec_conv3_weights
    weights_mem += 3 * sizeof(float);                   // dec_conv3_bias
    
    size_t gradients_mem = weights_mem;
    
    size_t activations_mem = 0;
    activations_mem += config.batch_size * 3 * 32 * 32 * sizeof(float);    // d_input
    activations_mem += config.batch_size * 256 * 32 * 32 * sizeof(float);  // d_enc_conv1_out
    activations_mem += config.batch_size * 256 * 16 * 16 * sizeof(float);  // d_enc_pool1_out
    activations_mem += config.batch_size * 128 * 16 * 16 * sizeof(float);  // d_enc_conv2_out
    activations_mem += config.batch_size * 128 * 8 * 8 * sizeof(float);    // d_enc_pool2_out
    activations_mem += config.batch_size * 128 * 8 * 8 * sizeof(float);    // d_dec_conv1_out
    activations_mem += config.batch_size * 128 * 16 * 16 * sizeof(float);  // d_dec_up1_out
    activations_mem += config.batch_size * 256 * 16 * 16 * sizeof(float);  // d_dec_conv2_out
    activations_mem += config.batch_size * 256 * 32 * 32 * sizeof(float);  // d_dec_up2_out
    activations_mem += config.batch_size * 3 * 32 * 32 * sizeof(float);    // d_dec_conv3_out
    activations_mem += config.batch_size * 3 * 32 * 32 * sizeof(float);    // d_target
    
    size_t grad_activations_mem = 0;
    grad_activations_mem += config.batch_size * 3 * 32 * 32 * sizeof(float);    // d_grad_output
    grad_activations_mem += config.batch_size * 3 * 32 * 32 * sizeof(float);    // d_grad_dec_conv3_out
    grad_activations_mem += config.batch_size * 256 * 32 * 32 * sizeof(float);  // d_grad_dec_up2_out
    grad_activations_mem += config.batch_size * 256 * 16 * 16 * sizeof(float);  // d_grad_dec_conv2_out
    grad_activations_mem += config.batch_size * 128 * 16 * 16 * sizeof(float);  // d_grad_dec_up1_out
    grad_activations_mem += config.batch_size * 128 * 8 * 8 * sizeof(float);    // d_grad_dec_conv1_out
    grad_activations_mem += config.batch_size * 128 * 8 * 8 * sizeof(float);    // d_grad_enc_pool2_out
    grad_activations_mem += config.batch_size * 128 * 16 * 16 * sizeof(float);  // d_grad_enc_conv2_out
    grad_activations_mem += config.batch_size * 256 * 16 * 16 * sizeof(float);  // d_grad_enc_pool1_out
    grad_activations_mem += config.batch_size * 256 * 32 * 32 * sizeof(float);  // d_grad_enc_conv1_out
    
    size_t masks_mem = 0;
    masks_mem += config.batch_size * 256 * 16 * 16 * sizeof(int);  // d_pool1_mask
    masks_mem += config.batch_size * 128 * 8 * 8 * sizeof(int);    // d_pool2_mask
    
    size_t total_model_mem = weights_mem + gradients_mem + activations_mem + grad_activations_mem + masks_mem;
    
    // Query actual GPU memory usage
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t used_mem = total_mem - free_mem;
    
    std::cout << "GPU Memory Usage (Model):" << std::endl;
    std::cout << "  Weights:              " << std::fixed << std::setprecision(2) << (weights_mem / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "  Gradients:            " << (gradients_mem / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "  Activations:          " << (activations_mem / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "  Gradient activations: " << (grad_activations_mem / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "  Pooling masks:        " << (masks_mem / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "  Total (Model):        " << (total_model_mem / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "GPU Device Memory:" << std::endl;
    std::cout << "  Total:                " << (total_mem / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "  Used:                 " << (used_mem / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "  Free:                 " << (free_mem / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << std::endl;
    
    // Create batch generator
    BatchGenerator batch_gen(dataset, config.batch_size, true, config.seed);
    
    // Allocate batch buffer
    float* batch_images = new float[config.batch_size * Constants::CIFAR_IMG_PIXELS];
    
    Timer total_timer("Total Training");
    
    // Training loop
    for (int epoch = 0; epoch < config.epochs; ++epoch) {
        Timer epoch_timer("Epoch");
        batch_gen.reset(true);
        
        float epoch_loss = 0.0f;
        int num_batches = 0;
        int batch_idx = 0;
        
        std::cout << "Starting epoch " << epoch + 1 << "/" << config.epochs << "..." << std::endl;
        
        while (batch_gen.has_next()) {
            int actual_batch_size = batch_gen.next_batch(batch_images);
            
            if (actual_batch_size < config.batch_size) {
                continue;
            }
            
            // Forward pass
            float loss = autoencoder.forward(batch_images, nullptr);
            
            // Backward pass
            autoencoder.backward(batch_images);
            
            // Update weights
            autoencoder.update_weights(config.learning_rate);
            
            epoch_loss += loss;
            num_batches++;
            batch_idx++;
            
            if (batch_idx % config.print_every == 0) {
                print_progress(batch_idx, batch_gen.num_batches(), epoch_loss / num_batches);
            }
        }
        
        // Synchronize at end of epoch for accurate timing
        cudaDeviceSynchronize();
        
        epoch_loss /= num_batches;
        double epoch_time = epoch_timer.elapsed();
        
        stats.epoch_losses.push_back(epoch_loss);
        stats.epoch_times.push_back(epoch_time);
        
        std::cout << std::endl;
        std::cout << "Epoch " << epoch + 1 << "/" << config.epochs 
                  << " - Loss: " << epoch_loss 
                  << " - Time: " << epoch_time << "s" << std::endl;
    }
    
    stats.total_time = total_timer.elapsed();
    stats.final_loss = stats.epoch_losses.back();
    
    // Save model
    autoencoder.save_weights(config.save_path);
    
    delete[] batch_images;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Training Complete!" << std::endl;
    std::cout << "Total time: " << stats.total_time << "s" << std::endl;
    std::cout << "Final loss: " << stats.final_loss << std::endl;
    std::cout << "========================================\n" << std::endl;
}

void train_gpu_optimized(CIFAR10Dataset& dataset, const TrainingConfig& config, TrainingStats& stats) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "GPU Training - OPTIMIZED v1 (Fused Kernels)" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Phase: GPU Opt v1" << std::endl;
    std::cout << "Optimizations:" << std::endl;
    std::cout << "  - Fused Conv+ReLU kernels" << std::endl;
    std::cout << "  - Loop unrolling (#pragma unroll)" << std::endl;
    std::cout << "  - Vectorized operations (float4)" << std::endl;
    std::cout << "  - __restrict__ pointers" << std::endl;
    std::cout << "  - Stride-1 fast path in backward" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Batch size: " << config.batch_size << std::endl;
    std::cout << "Epochs: " << config.epochs << std::endl;
    std::cout << "Learning rate: " << config.learning_rate << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // Initialize autoencoder with optimizations
    GPUAutoencoder autoencoder(true);
    autoencoder.initialize(config.batch_size, config.seed);
    
    // Calculate and display GPU memory usage
    size_t weights_mem = 0;
    weights_mem += 256 * 3 * 3 * 3 * sizeof(float);    // enc_conv1_weights
    weights_mem += 256 * sizeof(float);                 // enc_conv1_bias
    weights_mem += 128 * 256 * 3 * 3 * sizeof(float);  // enc_conv2_weights
    weights_mem += 128 * sizeof(float);                 // enc_conv2_bias
    weights_mem += 128 * 128 * 3 * 3 * sizeof(float);  // dec_conv1_weights
    weights_mem += 128 * sizeof(float);                 // dec_conv1_bias
    weights_mem += 256 * 128 * 3 * 3 * sizeof(float);  // dec_conv2_weights
    weights_mem += 256 * sizeof(float);                 // dec_conv2_bias
    weights_mem += 3 * 256 * 3 * 3 * sizeof(float);    // dec_conv3_weights
    weights_mem += 3 * sizeof(float);                   // dec_conv3_bias
    
    size_t gradients_mem = weights_mem;
    
    size_t activations_mem = 0;
    activations_mem += config.batch_size * 3 * 32 * 32 * sizeof(float);    // d_input
    activations_mem += config.batch_size * 256 * 32 * 32 * sizeof(float);  // d_enc_conv1_out
    activations_mem += config.batch_size * 256 * 16 * 16 * sizeof(float);  // d_enc_pool1_out
    activations_mem += config.batch_size * 128 * 16 * 16 * sizeof(float);  // d_enc_conv2_out
    activations_mem += config.batch_size * 128 * 8 * 8 * sizeof(float);    // d_enc_pool2_out
    activations_mem += config.batch_size * 128 * 8 * 8 * sizeof(float);    // d_dec_conv1_out
    activations_mem += config.batch_size * 128 * 16 * 16 * sizeof(float);  // d_dec_up1_out
    activations_mem += config.batch_size * 256 * 16 * 16 * sizeof(float);  // d_dec_conv2_out
    activations_mem += config.batch_size * 256 * 32 * 32 * sizeof(float);  // d_dec_up2_out
    activations_mem += config.batch_size * 3 * 32 * 32 * sizeof(float);    // d_dec_conv3_out
    activations_mem += config.batch_size * 3 * 32 * 32 * sizeof(float);    // d_target
    
    size_t grad_activations_mem = 0;
    grad_activations_mem += config.batch_size * 3 * 32 * 32 * sizeof(float);    // d_grad_output
    grad_activations_mem += config.batch_size * 3 * 32 * 32 * sizeof(float);    // d_grad_dec_conv3_out
    grad_activations_mem += config.batch_size * 256 * 32 * 32 * sizeof(float);  // d_grad_dec_up2_out
    grad_activations_mem += config.batch_size * 256 * 16 * 16 * sizeof(float);  // d_grad_dec_conv2_out
    grad_activations_mem += config.batch_size * 128 * 16 * 16 * sizeof(float);  // d_grad_dec_up1_out
    grad_activations_mem += config.batch_size * 128 * 8 * 8 * sizeof(float);    // d_grad_dec_conv1_out
    grad_activations_mem += config.batch_size * 128 * 8 * 8 * sizeof(float);    // d_grad_enc_pool2_out
    grad_activations_mem += config.batch_size * 128 * 16 * 16 * sizeof(float);  // d_grad_enc_conv2_out
    grad_activations_mem += config.batch_size * 256 * 16 * 16 * sizeof(float);  // d_grad_enc_pool1_out
    grad_activations_mem += config.batch_size * 256 * 32 * 32 * sizeof(float);  // d_grad_enc_conv1_out
    
    size_t masks_mem = 0;
    masks_mem += config.batch_size * 256 * 16 * 16 * sizeof(int);  // d_pool1_mask
    masks_mem += config.batch_size * 128 * 8 * 8 * sizeof(int);    // d_pool2_mask
    
    size_t total_model_mem = weights_mem + gradients_mem + activations_mem + grad_activations_mem + masks_mem;
    
    // Query actual GPU memory usage
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t used_mem = total_mem - free_mem;
    
    std::cout << "GPU Memory Usage (Model):" << std::endl;
    std::cout << "  Weights:              " << std::fixed << std::setprecision(2) << (weights_mem / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "  Gradients:            " << (gradients_mem / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "  Activations:          " << (activations_mem / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "  Gradient activations: " << (grad_activations_mem / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "  Pooling masks:        " << (masks_mem / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "  Total (Model):        " << (total_model_mem / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "GPU Device Memory:" << std::endl;
    std::cout << "  Total:                " << (total_mem / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "  Used:                 " << (used_mem / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "  Free:                 " << (free_mem / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << std::endl;
    
    BatchGenerator batch_gen(dataset, config.batch_size, true, config.seed);
    float* batch_images = new float[config.batch_size * Constants::CIFAR_IMG_PIXELS];
    
    Timer total_timer("Total Training");
    
    for (int epoch = 0; epoch < config.epochs; ++epoch) {
        Timer epoch_timer("Epoch");
        batch_gen.reset(true);
        
        float epoch_loss = 0.0f;
        int num_batches = 0;
        int batch_idx = 0;
        
        std::cout << "Starting epoch " << epoch + 1 << "/" << config.epochs << "..." << std::endl;
        
        while (batch_gen.has_next()) {
            int actual_batch_size = batch_gen.next_batch(batch_images);
            
            if (actual_batch_size < config.batch_size) {
                continue;
            }
            
            float loss = autoencoder.forward(batch_images, nullptr);
            autoencoder.backward(batch_images);
            autoencoder.update_weights(config.learning_rate);
            
            epoch_loss += loss;
            num_batches++;
            batch_idx++;
            
            if (batch_idx % config.print_every == 0) {
                print_progress(batch_idx, batch_gen.num_batches(), epoch_loss / num_batches);
            }
        }
        
        // Synchronize at end of epoch for accurate timing
        cudaDeviceSynchronize();
        
        epoch_loss /= num_batches;
        double epoch_time = epoch_timer.elapsed();
        
        stats.epoch_losses.push_back(epoch_loss);
        stats.epoch_times.push_back(epoch_time);
        
        std::cout << std::endl;
        std::cout << "Epoch " << epoch + 1 << "/" << config.epochs 
                  << " - Loss: " << epoch_loss 
                  << " - Time: " << epoch_time << "s" << std::endl;
    }
    
    stats.total_time = total_timer.elapsed();
    stats.final_loss = stats.epoch_losses.back();
    
    autoencoder.save_weights(config.save_path);
    
    delete[] batch_images;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Training Complete!" << std::endl;
    std::cout << "Total time: " << stats.total_time << "s" << std::endl;
    std::cout << "Final loss: " << stats.final_loss << std::endl;
    std::cout << "========================================\n" << std::endl;
}

void extract_features_gpu(CIFAR10Dataset& dataset, const std::string& model_path,
                          float* train_features, float* test_features, bool use_optimized) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "GPU Feature Extraction" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Mode: " << (use_optimized ? "Optimized" : "Basic") << std::endl;
    std::cout << "Train samples: " << Constants::CIFAR_TRAIN_SIZE << std::endl;
    std::cout << "Test samples: " << Constants::CIFAR_TEST_SIZE << std::endl;
    std::cout << "Feature dimension: " << Constants::FEATURE_DIM << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    Timer total_timer("Feature Extraction");
    
    const int batch_size = 100;
    
    GPUAutoencoder autoencoder(use_optimized);
    autoencoder.initialize(batch_size);
    
    if (!autoencoder.load_weights(model_path)) {
        std::cerr << "Error: Failed to load model weights" << std::endl;
        return;
    }
    
    float* batch_images = new float[batch_size * Constants::CIFAR_IMG_PIXELS];
    float* batch_features = new float[batch_size * Constants::FEATURE_DIM];
    
    // Extract training features
    Timer train_timer("Train Features");
    std::cout << "Extracting training features..." << std::endl;
    for (int i = 0; i < Constants::CIFAR_TRAIN_SIZE; i += batch_size) {
        int actual_batch = std::min(batch_size, Constants::CIFAR_TRAIN_SIZE - i);
        
        std::vector<int> indices(actual_batch);
        for (int j = 0; j < actual_batch; ++j) {
            indices[j] = i + j;
        }
        dataset.get_batch(batch_images, indices, true);
        
        autoencoder.encode(batch_images, batch_features);
        
        memcpy(train_features + i * Constants::FEATURE_DIM,
               batch_features, actual_batch * Constants::FEATURE_DIM * sizeof(float));
        
        if ((i / batch_size) % 50 == 0) {
            print_progress(i, Constants::CIFAR_TRAIN_SIZE);
        }
    }
    double train_time = train_timer.elapsed();
    std::cout << std::endl;
    std::cout << "Training features extracted in " << std::fixed << std::setprecision(2) << train_time << "s" << std::endl;
    
    // Extract test features
    Timer test_timer("Test Features");
    std::cout << "\nExtracting test features..." << std::endl;
    for (int i = 0; i < Constants::CIFAR_TEST_SIZE; i += batch_size) {
        int actual_batch = std::min(batch_size, Constants::CIFAR_TEST_SIZE - i);
        
        std::vector<int> indices(actual_batch);
        for (int j = 0; j < actual_batch; ++j) {
            indices[j] = i + j;
        }
        dataset.get_batch(batch_images, indices, false);
        
        autoencoder.encode(batch_images, batch_features);
        
        memcpy(test_features + i * Constants::FEATURE_DIM,
               batch_features, actual_batch * Constants::FEATURE_DIM * sizeof(float));
        
        if ((i / batch_size) % 10 == 0) {
            print_progress(i, Constants::CIFAR_TEST_SIZE);
        }
    }
    double test_time = test_timer.elapsed();
    std::cout << std::endl;
    std::cout << "Test features extracted in " << std::fixed << std::setprecision(2) << test_time << "s" << std::endl;
    
    delete[] batch_images;
    delete[] batch_features;
    
    double total_time = total_timer.elapsed();
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Feature Extraction Summary" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Train features time: " << std::fixed << std::setprecision(2) << train_time << "s" << std::endl;
    std::cout << "Test features time:  " << test_time << "s" << std::endl;
    std::cout << "Total time:          " << total_time << "s" << std::endl;
    std::cout << "Train features shape: (" << Constants::CIFAR_TRAIN_SIZE << ", " << Constants::FEATURE_DIM << ")" << std::endl;
    std::cout << "Test features shape:  (" << Constants::CIFAR_TEST_SIZE << ", " << Constants::FEATURE_DIM << ")" << std::endl;
    std::cout << "========================================\n" << std::endl;
}

// ============================================================================
// GPU Training Optimized v2: With CUDA Streams + Pinned Memory + Double Buffering
// ============================================================================

void train_gpu_optimized_v2(CIFAR10Dataset& dataset, const TrainingConfig& config, TrainingStats& stats) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "GPU Training (Optimized v2 - Streams + Pinned Memory)" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Batch size: " << config.batch_size << std::endl;
    std::cout << "Epochs: " << config.epochs << std::endl;
    std::cout << "Learning rate: " << config.learning_rate << std::endl;
    std::cout << "Optimizations:" << std::endl;
    std::cout << "  - Fused Conv+ReLU kernels" << std::endl;
    std::cout << "  - Vectorized operations (float4)" << std::endl;
    std::cout << "  - Pinned host memory (faster H2D)" << std::endl;
    std::cout << "  - CUDA streams (overlapped copy)" << std::endl;
    std::cout << "  - Double buffering" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // Create CUDA streams for overlapping computation and memory transfers
    cudaStream_t stream_compute, stream_transfer;
    CUDA_CHECK(cudaStreamCreate(&stream_compute));
    CUDA_CHECK(cudaStreamCreate(&stream_transfer));
    
    // Initialize autoencoder with optimizations
    GPUAutoencoder autoencoder(true);  // use_optimized = true
    autoencoder.initialize(config.batch_size, config.seed);
    
    BatchGenerator batch_gen(dataset, config.batch_size, true, config.seed);
    
    // Allocate pinned memory for faster host-device transfers (double buffering)
    float* h_batch_A;
    float* h_batch_B;
    CUDA_CHECK(cudaMallocHost(&h_batch_A, config.batch_size * Constants::CIFAR_IMG_PIXELS * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_batch_B, config.batch_size * Constants::CIFAR_IMG_PIXELS * sizeof(float)));
    
    std::cout << "Pinned memory allocated: " << (2 * config.batch_size * Constants::CIFAR_IMG_PIXELS * sizeof(float) / 1024.0) << " KB" << std::endl;
    std::cout << "CUDA Streams: 2 (compute + transfer)" << std::endl;
    std::cout << std::endl;
    
    Timer total_timer("Total Training");
    
    for (int epoch = 0; epoch < config.epochs; ++epoch) {
        Timer epoch_timer("Epoch");
        batch_gen.reset(true);
        
        float epoch_loss = 0.0f;
        int num_batches = 0;
        int batch_idx = 0;
        
        std::cout << "Starting epoch " << epoch + 1 << "/" << config.epochs << " (with streams)..." << std::endl;
        
        // Double buffering: A = current batch, B = next batch
        float* current_buffer = h_batch_A;
        float* next_buffer = h_batch_B;
        
        // Load first batch into current buffer
        bool has_current = batch_gen.has_next();
        int current_batch_size = 0;
        if (has_current) {
            current_batch_size = batch_gen.next_batch(current_buffer);
        }
        
        while (has_current && current_batch_size == config.batch_size) {
            // Start async copy of current batch to GPU
            autoencoder.copy_input_async(current_buffer, stream_transfer);
            
            // While GPU is receiving data, prepare next batch on CPU
            bool has_next = batch_gen.has_next();
            int next_batch_size = 0;
            if (has_next) {
                next_batch_size = batch_gen.next_batch(next_buffer);
            }
            
            // Run forward (waits for transfer to complete internally)
            float loss = autoencoder.forward_async(stream_compute);
            
            // Run backward
            autoencoder.backward_async(stream_compute);
            
            // Update weights
            autoencoder.update_weights_async(config.learning_rate, stream_compute);
            
            epoch_loss += loss;
            num_batches++;
            batch_idx++;
            
            if (batch_idx % config.print_every == 0) {
                print_progress(batch_idx, batch_gen.num_batches(), epoch_loss / num_batches);
            }
            
            // Swap buffers for next iteration
            std::swap(current_buffer, next_buffer);
            has_current = has_next;
            current_batch_size = next_batch_size;
        }
        
        // Synchronize at end of epoch for accurate timing
        CUDA_CHECK(cudaDeviceSynchronize());
        
        epoch_loss /= num_batches;
        double epoch_time = epoch_timer.elapsed();
        
        stats.epoch_losses.push_back(epoch_loss);
        stats.epoch_times.push_back(epoch_time);
        
        std::cout << std::endl;
        std::cout << "Epoch " << epoch + 1 << "/" << config.epochs 
                  << " - Loss: " << epoch_loss 
                  << " - Time: " << epoch_time << "s" << std::endl;
    }
    
    stats.total_time = total_timer.elapsed();
    stats.final_loss = stats.epoch_losses.back();
    
    autoencoder.save_weights(config.save_path);
    
    // Cleanup
    CUDA_CHECK(cudaFreeHost(h_batch_A));
    CUDA_CHECK(cudaFreeHost(h_batch_B));
    CUDA_CHECK(cudaStreamDestroy(stream_compute));
    CUDA_CHECK(cudaStreamDestroy(stream_transfer));
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Training Complete!" << std::endl;
    std::cout << "Total time: " << stats.total_time << "s" << std::endl;
    std::cout << "Final loss: " << stats.final_loss << std::endl;
    std::cout << "========================================\n" << std::endl;
}

