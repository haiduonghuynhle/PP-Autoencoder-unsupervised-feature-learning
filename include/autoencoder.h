#ifndef AUTOENCODER_H
#define AUTOENCODER_H

#include "utils.h"
#include "layers.h"
#include <string>

// Layer configuration structure
struct LayerConfig {
    int in_channels;
    int out_channels;
    int in_height;
    int in_width;
    int kernel_size;
    int stride;
    int padding;
};

// ============================================================================
// CPU Autoencoder
// ============================================================================

class CPUAutoencoder {
private:
    // Encoder weights and biases
    float* enc_conv1_weights;  // [256, 3, 3, 3]
    float* enc_conv1_bias;     // [256]
    float* enc_conv2_weights;  // [128, 256, 3, 3]
    float* enc_conv2_bias;     // [128]
    
    // Decoder weights and biases
    float* dec_conv1_weights;  // [128, 128, 3, 3]
    float* dec_conv1_bias;     // [128]
    float* dec_conv2_weights;  // [256, 128, 3, 3]
    float* dec_conv2_bias;     // [256]
    float* dec_conv3_weights;  // [3, 256, 3, 3]
    float* dec_conv3_bias;     // [3]
    
    // Gradients for weights
    float* grad_enc_conv1_weights;
    float* grad_enc_conv1_bias;
    float* grad_enc_conv2_weights;
    float* grad_enc_conv2_bias;
    float* grad_dec_conv1_weights;
    float* grad_dec_conv1_bias;
    float* grad_dec_conv2_weights;
    float* grad_dec_conv2_bias;
    float* grad_dec_conv3_weights;
    float* grad_dec_conv3_bias;
    
    // Intermediate activations (for forward pass)
    float* enc_conv1_out;      // [batch, 256, 32, 32]
    float* enc_relu1_out;      // [batch, 256, 32, 32]
    float* enc_pool1_out;      // [batch, 256, 16, 16]
    float* enc_conv2_out;      // [batch, 128, 16, 16]
    float* enc_relu2_out;      // [batch, 128, 16, 16]
    float* enc_pool2_out;      // [batch, 128, 8, 8] - LATENT
    
    float* dec_conv1_out;      // [batch, 128, 8, 8]
    float* dec_relu1_out;      // [batch, 128, 8, 8]
    float* dec_up1_out;        // [batch, 128, 16, 16]
    float* dec_conv2_out;      // [batch, 256, 16, 16]
    float* dec_relu2_out;      // [batch, 256, 16, 16]
    float* dec_up2_out;        // [batch, 256, 32, 32]
    float* dec_conv3_out;      // [batch, 3, 32, 32] - OUTPUT
    
    // Pooling masks (for backward pass)
    int* pool1_mask;
    int* pool2_mask;
    
    // Gradient buffers
    float* grad_dec_conv3_out;
    float* grad_dec_up2_out;
    float* grad_dec_relu2_out;
    float* grad_dec_conv2_out;
    float* grad_dec_up1_out;
    float* grad_dec_relu1_out;
    float* grad_dec_conv1_out;
    float* grad_enc_pool2_out;
    float* grad_enc_relu2_out;
    float* grad_enc_conv2_out;
    float* grad_enc_pool1_out;
    float* grad_enc_relu1_out;
    float* grad_enc_conv1_out;
    
    int batch_size;
    bool initialized;
    
    void allocate_memory();
    void free_memory();
    void zero_gradients();
    
public:
    CPUAutoencoder();
    ~CPUAutoencoder();
    
    // Initialize network with batch size
    void initialize(int batch_sz, unsigned int seed = 42);
    
    // Forward pass (full autoencoder)
    float forward(const float* input, float* output);
    
    // Encoder only (for feature extraction)
    void encode(const float* input, float* features);
    
    // Backward pass
    void backward(const float* input);
    
    // Update weights using SGD
    void update_weights(float learning_rate);
    
    // Save/Load model weights
    bool save_weights(const std::string& filepath) const;
    bool load_weights(const std::string& filepath);
    
    // Get latent features (after forward pass)
    const float* get_latent() const { return enc_pool2_out; }
    
    // Get output (after forward pass)
    const float* get_output() const { return dec_conv3_out; }
    
    int get_batch_size() const { return batch_size; }
};

// ============================================================================
// GPU Autoencoder (Naive Implementation)
// ============================================================================

#ifdef __CUDACC__

class GPUAutoencoder {
private:
    // Device pointers for weights
    float* d_enc_conv1_weights;
    float* d_enc_conv1_bias;
    float* d_enc_conv2_weights;
    float* d_enc_conv2_bias;
    float* d_dec_conv1_weights;
    float* d_dec_conv1_bias;
    float* d_dec_conv2_weights;
    float* d_dec_conv2_bias;
    float* d_dec_conv3_weights;
    float* d_dec_conv3_bias;
    
    // Device pointers for gradients
    float* d_grad_enc_conv1_weights;
    float* d_grad_enc_conv1_bias;
    float* d_grad_enc_conv2_weights;
    float* d_grad_enc_conv2_bias;
    float* d_grad_dec_conv1_weights;
    float* d_grad_dec_conv1_bias;
    float* d_grad_dec_conv2_weights;
    float* d_grad_dec_conv2_bias;
    float* d_grad_dec_conv3_weights;
    float* d_grad_dec_conv3_bias;
    
    // Device pointers for activations
    float* d_input;
    float* d_enc_conv1_out;
    float* d_enc_pool1_out;
    float* d_enc_conv2_out;
    float* d_enc_pool2_out;  // LATENT
    float* d_dec_conv1_out;
    float* d_dec_up1_out;
    float* d_dec_conv2_out;
    float* d_dec_up2_out;
    float* d_dec_conv3_out;  // OUTPUT
    float* d_target;
    
    // Pooling masks
    int* d_pool1_mask;
    int* d_pool2_mask;
    
    // Gradient activations
    float* d_grad_output;
    float* d_grad_dec_conv3_out;
    float* d_grad_dec_up2_out;
    float* d_grad_dec_conv2_out;
    float* d_grad_dec_up1_out;
    float* d_grad_dec_conv1_out;
    float* d_grad_enc_pool2_out;
    float* d_grad_enc_conv2_out;
    float* d_grad_enc_pool1_out;
    float* d_grad_enc_conv1_out;
    
    // Host buffers for initialization
    float* h_latent;
    float* h_output;
    
    int batch_size;
    bool initialized;
    bool use_optimized;
    
    void allocate_device_memory();
    void free_device_memory();
    void zero_gradients();
    
public:
    GPUAutoencoder(bool optimized = false);
    ~GPUAutoencoder();
    
    // Initialize with random weights
    void initialize(int batch_sz, unsigned int seed = 42);
    
    // Initialize from CPU autoencoder (copy weights)
    void initialize_from_cpu(const CPUAutoencoder& cpu_ae, int batch_sz);
    
    // Copy input to device and run forward pass
    float forward(const float* h_input, float* h_output = nullptr);
    
    // Encoder only (for feature extraction)
    void encode(const float* h_input, float* h_features);
    
    // Backward pass (input should already be on device from forward)
    void backward(const float* h_target);
    
    // Update weights
    void update_weights(float learning_rate);
    
    // Copy weights back to host
    void copy_weights_to_host(CPUAutoencoder& cpu_ae) const;
    
    // Save/Load model
    bool save_weights(const std::string& filepath) const;
    bool load_weights(const std::string& filepath);
    
    // Get latent features (copies from device)
    void get_latent(float* h_features) const;
    
    int get_batch_size() const { return batch_size; }
    
    // Set optimization mode
    void set_optimized(bool opt) { use_optimized = opt; }
};

#endif // __CUDACC__

// ============================================================================
// Training utilities
// ============================================================================

// Optimization levels
enum class OptLevel {
    GPU_BASIC = 0,    // Naive parallelization
    GPU_OPT_V1 = 1,   // Shared memory tiling
    GPU_OPT_V2 = 2    // Kernel fusion + CUDA streams
};

struct TrainingConfig {
    int batch_size = 64;
    int epochs = 20;
    float learning_rate = 0.001f;
    bool use_gpu = true;
    bool use_optimized = false;  // Legacy: true = opt_v1
    OptLevel opt_level = OptLevel::GPU_BASIC;
    std::string save_path = "models/autoencoder.bin";
    int print_every = 100;  // Print loss every N batches
    unsigned int seed = 42;
};

// Training statistics
struct TrainingStats {
    std::vector<float> epoch_losses;
    std::vector<double> epoch_times;
    double total_time = 0.0;
    float final_loss = 0.0f;
};

#endif // AUTOENCODER_H

