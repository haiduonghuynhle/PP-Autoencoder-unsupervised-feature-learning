#ifndef LAYERS_H
#define LAYERS_H

#include "utils.h"

// ============================================================================
// CPU Layer Implementations
// ============================================================================

namespace cpu {

// Conv2D forward pass
// input: [batch, in_channels, height, width]
// weights: [out_channels, in_channels, kernel_h, kernel_w]
// bias: [out_channels]
// output: [batch, out_channels, out_height, out_width]
void conv2d_forward(
    const float* input, const float* weights, const float* bias,
    float* output,
    int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int kernel_size, int stride, int padding
);

// Conv2D backward pass
void conv2d_backward(
    const float* input, const float* weights, const float* grad_output,
    float* grad_input, float* grad_weights, float* grad_bias,
    int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int kernel_size, int stride, int padding
);

// ReLU forward pass (in-place capable)
void relu_forward(const float* input, float* output, int size);

// ReLU backward pass
void relu_backward(const float* input, const float* grad_output, 
                   float* grad_input, int size);

// MaxPool2D forward pass
// input: [batch, channels, height, width]
// output: [batch, channels, height/2, width/2]
// mask: stores indices for backward pass
void maxpool2d_forward(
    const float* input, float* output, int* mask,
    int batch_size, int channels, int in_height, int in_width,
    int pool_size, int stride
);

// MaxPool2D backward pass
void maxpool2d_backward(
    const float* grad_output, const int* mask, float* grad_input,
    int batch_size, int channels, int in_height, int in_width,
    int pool_size, int stride
);

// Upsample2D forward pass (nearest neighbor)
// input: [batch, channels, height, width]
// output: [batch, channels, height*scale, width*scale]
void upsample2d_forward(
    const float* input, float* output,
    int batch_size, int channels, int in_height, int in_width,
    int scale_factor
);

// Upsample2D backward pass
void upsample2d_backward(
    const float* grad_output, float* grad_input,
    int batch_size, int channels, int in_height, int in_width,
    int scale_factor
);

// MSE Loss forward
float mse_loss_forward(const float* predicted, const float* target, int size);

// MSE Loss backward
void mse_loss_backward(const float* predicted, const float* target,
                       float* grad_output, int size);

// SGD weight update
void sgd_update(float* weights, const float* gradients, 
                float learning_rate, int size);

// Initialize weights using He initialization
void initialize_weights_he(float* weights, int fan_in, int size, unsigned int seed = 42);

// Initialize bias to zero
void initialize_bias_zero(float* bias, int size);

} // namespace cpu

// ============================================================================
// GPU Layer Implementations (Naive)
// ============================================================================

#ifdef __CUDACC__

namespace gpu {

// Conv2D forward pass
void conv2d_forward(
    const float* d_input, const float* d_weights, const float* d_bias,
    float* d_output,
    int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int kernel_size, int stride, int padding
);

// Conv2D backward pass
void conv2d_backward(
    const float* d_input, const float* d_weights, const float* d_grad_output,
    float* d_grad_input, float* d_grad_weights, float* d_grad_bias,
    int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int kernel_size, int stride, int padding
);

// ReLU forward pass
void relu_forward(const float* d_input, float* d_output, int size);

// ReLU backward pass
void relu_backward(const float* d_input, const float* d_grad_output,
                   float* d_grad_input, int size);

// MaxPool2D forward pass
void maxpool2d_forward(
    const float* d_input, float* d_output, int* d_mask,
    int batch_size, int channels, int in_height, int in_width,
    int pool_size, int stride
);

// MaxPool2D backward pass
void maxpool2d_backward(
    const float* d_grad_output, const int* d_mask, float* d_grad_input,
    int batch_size, int channels, int in_height, int in_width,
    int pool_size, int stride
);

// Upsample2D forward pass
void upsample2d_forward(
    const float* d_input, float* d_output,
    int batch_size, int channels, int in_height, int in_width,
    int scale_factor
);

// Upsample2D backward pass
void upsample2d_backward(
    const float* d_grad_output, float* d_grad_input,
    int batch_size, int channels, int in_height, int in_width,
    int scale_factor
);

// MSE Loss forward
float mse_loss_forward(const float* d_predicted, const float* d_target, int size);

// MSE Loss backward
void mse_loss_backward(const float* d_predicted, const float* d_target,
                       float* d_grad_output, int size);

// SGD weight update
void sgd_update(float* d_weights, const float* d_gradients,
                float learning_rate, int size);

} // namespace gpu

// ============================================================================
// GPU Optimized Layer Implementations
// ============================================================================

namespace gpu_optimized {

// Optimized Conv2D with shared memory tiling
void conv2d_forward_tiled(
    const float* d_input, const float* d_weights, const float* d_bias,
    float* d_output,
    int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int kernel_size, int stride, int padding
);

// Fused Conv2D + ReLU + Bias
void conv2d_relu_forward(
    const float* d_input, const float* d_weights, const float* d_bias,
    float* d_output,
    int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int kernel_size, int stride, int padding
);

// Optimized Conv2D backward with shared memory
void conv2d_backward_tiled(
    const float* d_input, const float* d_weights, const float* d_grad_output,
    float* d_grad_input, float* d_grad_weights, float* d_grad_bias,
    int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int kernel_size, int stride, int padding
);

// Vectorized ReLU using float4
void relu_forward_vectorized(const float* d_input, float* d_output, int size);

// Vectorized ReLU backward
void relu_backward_vectorized(const float* d_input, const float* d_grad_output,
                              float* d_grad_input, int size);

// Fused MaxPool + ReLU (for encoder)
void maxpool2d_forward_fused(
    const float* d_input, float* d_output, int* d_mask,
    int batch_size, int channels, int in_height, int in_width,
    int pool_size, int stride
);

// Optimized Upsample with coalesced memory access
void upsample2d_forward_coalesced(
    const float* d_input, float* d_output,
    int batch_size, int channels, int in_height, int in_width,
    int scale_factor
);

// Vectorized SGD update
void sgd_update_vectorized(float* d_weights, const float* d_gradients,
                           float learning_rate, int size);

} // namespace gpu_optimized

#endif // __CUDACC__

#endif // LAYERS_H

