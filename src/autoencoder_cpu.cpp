#include "autoencoder.h"
#include <fstream>
#include <cstring>

// ============================================================================
// CPUAutoencoder Implementation
// ============================================================================

CPUAutoencoder::CPUAutoencoder() 
    : enc_conv1_weights(nullptr), enc_conv1_bias(nullptr),
      enc_conv2_weights(nullptr), enc_conv2_bias(nullptr),
      dec_conv1_weights(nullptr), dec_conv1_bias(nullptr),
      dec_conv2_weights(nullptr), dec_conv2_bias(nullptr),
      dec_conv3_weights(nullptr), dec_conv3_bias(nullptr),
      batch_size(0), initialized(false) {
}

CPUAutoencoder::~CPUAutoencoder() {
    free_memory();
}

void CPUAutoencoder::allocate_memory() {
    // Calculate sizes
    // Encoder conv1: [256, 3, 3, 3]
    int enc_conv1_w_size = 256 * 3 * 3 * 3;
    // Encoder conv2: [128, 256, 3, 3]
    int enc_conv2_w_size = 128 * 256 * 3 * 3;
    // Decoder conv1: [128, 128, 3, 3]
    int dec_conv1_w_size = 128 * 128 * 3 * 3;
    // Decoder conv2: [256, 128, 3, 3]
    int dec_conv2_w_size = 256 * 128 * 3 * 3;
    // Decoder conv3: [3, 256, 3, 3]
    int dec_conv3_w_size = 3 * 256 * 3 * 3;
    
    // Allocate weights
    enc_conv1_weights = new float[enc_conv1_w_size];
    enc_conv1_bias = new float[256];
    enc_conv2_weights = new float[enc_conv2_w_size];
    enc_conv2_bias = new float[128];
    
    dec_conv1_weights = new float[dec_conv1_w_size];
    dec_conv1_bias = new float[128];
    dec_conv2_weights = new float[dec_conv2_w_size];
    dec_conv2_bias = new float[256];
    dec_conv3_weights = new float[dec_conv3_w_size];
    dec_conv3_bias = new float[3];
    
    // Allocate gradients
    grad_enc_conv1_weights = new float[enc_conv1_w_size];
    grad_enc_conv1_bias = new float[256];
    grad_enc_conv2_weights = new float[enc_conv2_w_size];
    grad_enc_conv2_bias = new float[128];
    
    grad_dec_conv1_weights = new float[dec_conv1_w_size];
    grad_dec_conv1_bias = new float[128];
    grad_dec_conv2_weights = new float[dec_conv2_w_size];
    grad_dec_conv2_bias = new float[256];
    grad_dec_conv3_weights = new float[dec_conv3_w_size];
    grad_dec_conv3_bias = new float[3];
    
    // Allocate activations (based on batch size)
    enc_conv1_out = new float[batch_size * 256 * 32 * 32];
    enc_relu1_out = new float[batch_size * 256 * 32 * 32];
    enc_pool1_out = new float[batch_size * 256 * 16 * 16];
    enc_conv2_out = new float[batch_size * 128 * 16 * 16];
    enc_relu2_out = new float[batch_size * 128 * 16 * 16];
    enc_pool2_out = new float[batch_size * 128 * 8 * 8];  // LATENT
    
    dec_conv1_out = new float[batch_size * 128 * 8 * 8];
    dec_relu1_out = new float[batch_size * 128 * 8 * 8];
    dec_up1_out = new float[batch_size * 128 * 16 * 16];
    dec_conv2_out = new float[batch_size * 256 * 16 * 16];
    dec_relu2_out = new float[batch_size * 256 * 16 * 16];
    dec_up2_out = new float[batch_size * 256 * 32 * 32];
    dec_conv3_out = new float[batch_size * 3 * 32 * 32];  // OUTPUT
    
    // Allocate pooling masks
    pool1_mask = new int[batch_size * 256 * 16 * 16];
    pool2_mask = new int[batch_size * 128 * 8 * 8];
    
    // Allocate gradient buffers
    grad_dec_conv3_out = new float[batch_size * 3 * 32 * 32];
    grad_dec_up2_out = new float[batch_size * 256 * 32 * 32];
    grad_dec_relu2_out = new float[batch_size * 256 * 16 * 16];
    grad_dec_conv2_out = new float[batch_size * 256 * 16 * 16];
    grad_dec_up1_out = new float[batch_size * 128 * 16 * 16];
    grad_dec_relu1_out = new float[batch_size * 128 * 8 * 8];
    grad_dec_conv1_out = new float[batch_size * 128 * 8 * 8];
    grad_enc_pool2_out = new float[batch_size * 128 * 8 * 8];
    grad_enc_relu2_out = new float[batch_size * 128 * 16 * 16];
    grad_enc_conv2_out = new float[batch_size * 128 * 16 * 16];
    grad_enc_pool1_out = new float[batch_size * 256 * 16 * 16];
    grad_enc_relu1_out = new float[batch_size * 256 * 32 * 32];
    grad_enc_conv1_out = new float[batch_size * 256 * 32 * 32];
}

void CPUAutoencoder::free_memory() {
    if (!initialized) return;
    
    // Free weights
    delete[] enc_conv1_weights;
    delete[] enc_conv1_bias;
    delete[] enc_conv2_weights;
    delete[] enc_conv2_bias;
    delete[] dec_conv1_weights;
    delete[] dec_conv1_bias;
    delete[] dec_conv2_weights;
    delete[] dec_conv2_bias;
    delete[] dec_conv3_weights;
    delete[] dec_conv3_bias;
    
    // Free gradients
    delete[] grad_enc_conv1_weights;
    delete[] grad_enc_conv1_bias;
    delete[] grad_enc_conv2_weights;
    delete[] grad_enc_conv2_bias;
    delete[] grad_dec_conv1_weights;
    delete[] grad_dec_conv1_bias;
    delete[] grad_dec_conv2_weights;
    delete[] grad_dec_conv2_bias;
    delete[] grad_dec_conv3_weights;
    delete[] grad_dec_conv3_bias;
    
    // Free activations
    delete[] enc_conv1_out;
    delete[] enc_relu1_out;
    delete[] enc_pool1_out;
    delete[] enc_conv2_out;
    delete[] enc_relu2_out;
    delete[] enc_pool2_out;
    delete[] dec_conv1_out;
    delete[] dec_relu1_out;
    delete[] dec_up1_out;
    delete[] dec_conv2_out;
    delete[] dec_relu2_out;
    delete[] dec_up2_out;
    delete[] dec_conv3_out;
    
    // Free masks
    delete[] pool1_mask;
    delete[] pool2_mask;
    
    // Free gradient buffers
    delete[] grad_dec_conv3_out;
    delete[] grad_dec_up2_out;
    delete[] grad_dec_relu2_out;
    delete[] grad_dec_conv2_out;
    delete[] grad_dec_up1_out;
    delete[] grad_dec_relu1_out;
    delete[] grad_dec_conv1_out;
    delete[] grad_enc_pool2_out;
    delete[] grad_enc_relu2_out;
    delete[] grad_enc_conv2_out;
    delete[] grad_enc_pool1_out;
    delete[] grad_enc_relu1_out;
    delete[] grad_enc_conv1_out;
    
    initialized = false;
}

void CPUAutoencoder::zero_gradients() {
    // Zero weight gradients
    memset(grad_enc_conv1_weights, 0, 256 * 3 * 3 * 3 * sizeof(float));
    memset(grad_enc_conv1_bias, 0, 256 * sizeof(float));
    memset(grad_enc_conv2_weights, 0, 128 * 256 * 3 * 3 * sizeof(float));
    memset(grad_enc_conv2_bias, 0, 128 * sizeof(float));
    memset(grad_dec_conv1_weights, 0, 128 * 128 * 3 * 3 * sizeof(float));
    memset(grad_dec_conv1_bias, 0, 128 * sizeof(float));
    memset(grad_dec_conv2_weights, 0, 256 * 128 * 3 * 3 * sizeof(float));
    memset(grad_dec_conv2_bias, 0, 256 * sizeof(float));
    memset(grad_dec_conv3_weights, 0, 3 * 256 * 3 * 3 * sizeof(float));
    memset(grad_dec_conv3_bias, 0, 3 * sizeof(float));
}

void CPUAutoencoder::initialize(int batch_sz, unsigned int seed) {
    if (initialized) {
        free_memory();
    }
    
    batch_size = batch_sz;
    allocate_memory();
    
    // Initialize weights using He initialization
    // fan_in = in_channels * kernel_size * kernel_size
    cpu::initialize_weights_he(enc_conv1_weights, 3 * 3 * 3, 256 * 3 * 3 * 3, seed);
    cpu::initialize_bias_zero(enc_conv1_bias, 256);
    
    cpu::initialize_weights_he(enc_conv2_weights, 256 * 3 * 3, 128 * 256 * 3 * 3, seed + 1);
    cpu::initialize_bias_zero(enc_conv2_bias, 128);
    
    cpu::initialize_weights_he(dec_conv1_weights, 128 * 3 * 3, 128 * 128 * 3 * 3, seed + 2);
    cpu::initialize_bias_zero(dec_conv1_bias, 128);
    
    cpu::initialize_weights_he(dec_conv2_weights, 128 * 3 * 3, 256 * 128 * 3 * 3, seed + 3);
    cpu::initialize_bias_zero(dec_conv2_bias, 256);
    
    cpu::initialize_weights_he(dec_conv3_weights, 256 * 3 * 3, 3 * 256 * 3 * 3, seed + 4);
    cpu::initialize_bias_zero(dec_conv3_bias, 3);
    
    initialized = true;
    std::cout << "CPU Autoencoder initialized with batch size " << batch_size << std::endl;
}

float CPUAutoencoder::forward(const float* input, float* output) {
    // ==================== ENCODER ====================
    
    // Conv1: (batch, 3, 32, 32) -> (batch, 256, 32, 32)
    cpu::conv2d_forward(input, enc_conv1_weights, enc_conv1_bias, enc_conv1_out,
                        batch_size, 3, 32, 32, 256, 3, 1, 1);
    
    // ReLU1
    cpu::relu_forward(enc_conv1_out, enc_relu1_out, batch_size * 256 * 32 * 32);
    
    // MaxPool1: (batch, 256, 32, 32) -> (batch, 256, 16, 16)
    cpu::maxpool2d_forward(enc_relu1_out, enc_pool1_out, pool1_mask,
                           batch_size, 256, 32, 32, 2, 2);
    
    // Conv2: (batch, 256, 16, 16) -> (batch, 128, 16, 16)
    cpu::conv2d_forward(enc_pool1_out, enc_conv2_weights, enc_conv2_bias, enc_conv2_out,
                        batch_size, 256, 16, 16, 128, 3, 1, 1);
    
    // ReLU2
    cpu::relu_forward(enc_conv2_out, enc_relu2_out, batch_size * 128 * 16 * 16);
    
    // MaxPool2: (batch, 128, 16, 16) -> (batch, 128, 8, 8) = LATENT
    cpu::maxpool2d_forward(enc_relu2_out, enc_pool2_out, pool2_mask,
                           batch_size, 128, 16, 16, 2, 2);
    
    // ==================== DECODER ====================
    
    // Conv3: (batch, 128, 8, 8) -> (batch, 128, 8, 8)
    cpu::conv2d_forward(enc_pool2_out, dec_conv1_weights, dec_conv1_bias, dec_conv1_out,
                        batch_size, 128, 8, 8, 128, 3, 1, 1);
    
    // ReLU3
    cpu::relu_forward(dec_conv1_out, dec_relu1_out, batch_size * 128 * 8 * 8);
    
    // Upsample1: (batch, 128, 8, 8) -> (batch, 128, 16, 16)
    cpu::upsample2d_forward(dec_relu1_out, dec_up1_out,
                            batch_size, 128, 8, 8, 2);
    
    // Conv4: (batch, 128, 16, 16) -> (batch, 256, 16, 16)
    cpu::conv2d_forward(dec_up1_out, dec_conv2_weights, dec_conv2_bias, dec_conv2_out,
                        batch_size, 128, 16, 16, 256, 3, 1, 1);
    
    // ReLU4
    cpu::relu_forward(dec_conv2_out, dec_relu2_out, batch_size * 256 * 16 * 16);
    
    // Upsample2: (batch, 256, 16, 16) -> (batch, 256, 32, 32)
    cpu::upsample2d_forward(dec_relu2_out, dec_up2_out,
                            batch_size, 256, 16, 16, 2);
    
    // Conv5: (batch, 256, 32, 32) -> (batch, 3, 32, 32) = OUTPUT
    cpu::conv2d_forward(dec_up2_out, dec_conv3_weights, dec_conv3_bias, dec_conv3_out,
                        batch_size, 256, 32, 32, 3, 3, 1, 1);
    
    // Copy output if requested
    if (output != nullptr) {
        memcpy(output, dec_conv3_out, batch_size * 3 * 32 * 32 * sizeof(float));
    }
    
    // Compute and return loss (MSE between input and reconstructed output)
    return cpu::mse_loss_forward(dec_conv3_out, input, batch_size * 3 * 32 * 32);
}

void CPUAutoencoder::encode(const float* input, float* features) {
    // Only run encoder part
    cpu::conv2d_forward(input, enc_conv1_weights, enc_conv1_bias, enc_conv1_out,
                        batch_size, 3, 32, 32, 256, 3, 1, 1);
    cpu::relu_forward(enc_conv1_out, enc_relu1_out, batch_size * 256 * 32 * 32);
    cpu::maxpool2d_forward(enc_relu1_out, enc_pool1_out, pool1_mask,
                           batch_size, 256, 32, 32, 2, 2);
    
    cpu::conv2d_forward(enc_pool1_out, enc_conv2_weights, enc_conv2_bias, enc_conv2_out,
                        batch_size, 256, 16, 16, 128, 3, 1, 1);
    cpu::relu_forward(enc_conv2_out, enc_relu2_out, batch_size * 128 * 16 * 16);
    cpu::maxpool2d_forward(enc_relu2_out, enc_pool2_out, pool2_mask,
                           batch_size, 128, 16, 16, 2, 2);
    
    // Copy latent features
    memcpy(features, enc_pool2_out, batch_size * 128 * 8 * 8 * sizeof(float));
}

void CPUAutoencoder::backward(const float* input) {
    zero_gradients();
    
    // Compute loss gradient: d(MSE)/d(output)
    cpu::mse_loss_backward(dec_conv3_out, input, grad_dec_conv3_out,
                           batch_size * 3 * 32 * 32);
    
    // ==================== DECODER BACKWARD ====================
    
    // Conv5 backward
    cpu::conv2d_backward(dec_up2_out, dec_conv3_weights, grad_dec_conv3_out,
                         grad_dec_up2_out, grad_dec_conv3_weights, grad_dec_conv3_bias,
                         batch_size, 256, 32, 32, 3, 3, 1, 1);
    
    // Upsample2 backward
    cpu::upsample2d_backward(grad_dec_up2_out, grad_dec_relu2_out,
                             batch_size, 256, 16, 16, 2);
    
    // ReLU4 backward
    cpu::relu_backward(dec_conv2_out, grad_dec_relu2_out, grad_dec_conv2_out,
                       batch_size * 256 * 16 * 16);
    
    // Conv4 backward
    cpu::conv2d_backward(dec_up1_out, dec_conv2_weights, grad_dec_conv2_out,
                         grad_dec_up1_out, grad_dec_conv2_weights, grad_dec_conv2_bias,
                         batch_size, 128, 16, 16, 256, 3, 1, 1);
    
    // Upsample1 backward
    cpu::upsample2d_backward(grad_dec_up1_out, grad_dec_relu1_out,
                             batch_size, 128, 8, 8, 2);
    
    // ReLU3 backward
    cpu::relu_backward(dec_conv1_out, grad_dec_relu1_out, grad_dec_conv1_out,
                       batch_size * 128 * 8 * 8);
    
    // Conv3 backward
    cpu::conv2d_backward(enc_pool2_out, dec_conv1_weights, grad_dec_conv1_out,
                         grad_enc_pool2_out, grad_dec_conv1_weights, grad_dec_conv1_bias,
                         batch_size, 128, 8, 8, 128, 3, 1, 1);
    
    // ==================== ENCODER BACKWARD ====================
    
    // MaxPool2 backward
    cpu::maxpool2d_backward(grad_enc_pool2_out, pool2_mask, grad_enc_relu2_out,
                            batch_size, 128, 16, 16, 2, 2);
    
    // ReLU2 backward
    cpu::relu_backward(enc_conv2_out, grad_enc_relu2_out, grad_enc_conv2_out,
                       batch_size * 128 * 16 * 16);
    
    // Conv2 backward
    cpu::conv2d_backward(enc_pool1_out, enc_conv2_weights, grad_enc_conv2_out,
                         grad_enc_pool1_out, grad_enc_conv2_weights, grad_enc_conv2_bias,
                         batch_size, 256, 16, 16, 128, 3, 1, 1);
    
    // MaxPool1 backward
    cpu::maxpool2d_backward(grad_enc_pool1_out, pool1_mask, grad_enc_relu1_out,
                            batch_size, 256, 32, 32, 2, 2);
    
    // ReLU1 backward
    cpu::relu_backward(enc_conv1_out, grad_enc_relu1_out, grad_enc_conv1_out,
                       batch_size * 256 * 32 * 32);
    
    // Conv1 backward (we don't need grad_input for the first layer)
    float* dummy_grad_input = new float[batch_size * 3 * 32 * 32];
    cpu::conv2d_backward(input, enc_conv1_weights, grad_enc_conv1_out,
                         dummy_grad_input, grad_enc_conv1_weights, grad_enc_conv1_bias,
                         batch_size, 3, 32, 32, 256, 3, 1, 1);
    delete[] dummy_grad_input;
}

void CPUAutoencoder::update_weights(float learning_rate) {
    // Update encoder weights
    cpu::sgd_update(enc_conv1_weights, grad_enc_conv1_weights, learning_rate, 256 * 3 * 3 * 3);
    cpu::sgd_update(enc_conv1_bias, grad_enc_conv1_bias, learning_rate, 256);
    cpu::sgd_update(enc_conv2_weights, grad_enc_conv2_weights, learning_rate, 128 * 256 * 3 * 3);
    cpu::sgd_update(enc_conv2_bias, grad_enc_conv2_bias, learning_rate, 128);
    
    // Update decoder weights
    cpu::sgd_update(dec_conv1_weights, grad_dec_conv1_weights, learning_rate, 128 * 128 * 3 * 3);
    cpu::sgd_update(dec_conv1_bias, grad_dec_conv1_bias, learning_rate, 128);
    cpu::sgd_update(dec_conv2_weights, grad_dec_conv2_weights, learning_rate, 256 * 128 * 3 * 3);
    cpu::sgd_update(dec_conv2_bias, grad_dec_conv2_bias, learning_rate, 256);
    cpu::sgd_update(dec_conv3_weights, grad_dec_conv3_weights, learning_rate, 3 * 256 * 3 * 3);
    cpu::sgd_update(dec_conv3_bias, grad_dec_conv3_bias, learning_rate, 3);
}

bool CPUAutoencoder::save_weights(const std::string& filepath) const {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file for writing: " << filepath << std::endl;
        return false;
    }
    
    // Write magic number and version
    int magic = 0xAE2024;
    int version = 1;
    file.write(reinterpret_cast<const char*>(&magic), sizeof(int));
    file.write(reinterpret_cast<const char*>(&version), sizeof(int));
    
    // Write weights
    file.write(reinterpret_cast<const char*>(enc_conv1_weights), 256 * 3 * 3 * 3 * sizeof(float));
    file.write(reinterpret_cast<const char*>(enc_conv1_bias), 256 * sizeof(float));
    file.write(reinterpret_cast<const char*>(enc_conv2_weights), 128 * 256 * 3 * 3 * sizeof(float));
    file.write(reinterpret_cast<const char*>(enc_conv2_bias), 128 * sizeof(float));
    file.write(reinterpret_cast<const char*>(dec_conv1_weights), 128 * 128 * 3 * 3 * sizeof(float));
    file.write(reinterpret_cast<const char*>(dec_conv1_bias), 128 * sizeof(float));
    file.write(reinterpret_cast<const char*>(dec_conv2_weights), 256 * 128 * 3 * 3 * sizeof(float));
    file.write(reinterpret_cast<const char*>(dec_conv2_bias), 256 * sizeof(float));
    file.write(reinterpret_cast<const char*>(dec_conv3_weights), 3 * 256 * 3 * 3 * sizeof(float));
    file.write(reinterpret_cast<const char*>(dec_conv3_bias), 3 * sizeof(float));
    
    file.close();
    std::cout << "Weights saved to " << filepath << std::endl;
    return true;
}

bool CPUAutoencoder::load_weights(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file for reading: " << filepath << std::endl;
        return false;
    }
    
    // Check magic number and version
    int magic, version;
    file.read(reinterpret_cast<char*>(&magic), sizeof(int));
    file.read(reinterpret_cast<char*>(&version), sizeof(int));
    
    if (magic != 0xAE2024) {
        std::cerr << "Error: Invalid model file format" << std::endl;
        return false;
    }
    
    // Read weights
    file.read(reinterpret_cast<char*>(enc_conv1_weights), 256 * 3 * 3 * 3 * sizeof(float));
    file.read(reinterpret_cast<char*>(enc_conv1_bias), 256 * sizeof(float));
    file.read(reinterpret_cast<char*>(enc_conv2_weights), 128 * 256 * 3 * 3 * sizeof(float));
    file.read(reinterpret_cast<char*>(enc_conv2_bias), 128 * sizeof(float));
    file.read(reinterpret_cast<char*>(dec_conv1_weights), 128 * 128 * 3 * 3 * sizeof(float));
    file.read(reinterpret_cast<char*>(dec_conv1_bias), 128 * sizeof(float));
    file.read(reinterpret_cast<char*>(dec_conv2_weights), 256 * 128 * 3 * 3 * sizeof(float));
    file.read(reinterpret_cast<char*>(dec_conv2_bias), 256 * sizeof(float));
    file.read(reinterpret_cast<char*>(dec_conv3_weights), 3 * 256 * 3 * 3 * sizeof(float));
    file.read(reinterpret_cast<char*>(dec_conv3_bias), 3 * sizeof(float));
    
    file.close();
    std::cout << "Weights loaded from " << filepath << std::endl;
    return true;
}

