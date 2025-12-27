#include "layers.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>

namespace gpu {

// ============================================================================
// CUDA Kernels - Naive Implementation
// ============================================================================

// Conv2D Forward Kernel
// Each thread computes one output pixel
__global__ void conv2d_forward_kernel(
    const float* input, const float* weights, const float* bias,
    float* output,
    int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int kernel_size, int stride, int padding,
    int out_height, int out_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * out_channels * out_height * out_width;
    
    if (idx >= total_outputs) return;
    
    // Decode linear index to (n, oc, oh, ow)
    int ow = idx % out_width;
    int oh = (idx / out_width) % out_height;
    int oc = (idx / (out_width * out_height)) % out_channels;
    int n = idx / (out_width * out_height * out_channels);
    
    float sum = 0.0f;
    
    // Convolution
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int ih = oh * stride - padding + kh;
                int iw = ow * stride - padding + kw;
                
                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                    int input_idx = ((n * in_channels + ic) * in_height + ih) * in_width + iw;
                    int weight_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                    sum += input[input_idx] * weights[weight_idx];
                }
            }
        }
    }
    
    // Add bias
    sum += bias[oc];
    output[idx] = sum;
}

// Conv2D Backward Kernel - Weight Gradients
__global__ void conv2d_backward_weights_kernel(
    const float* input, const float* grad_output,
    float* grad_weights, float* grad_bias,
    int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int kernel_size, int stride, int padding,
    int out_height, int out_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_weights = out_channels * in_channels * kernel_size * kernel_size;
    
    if (idx >= total_weights) return;
    
    // Decode to (oc, ic, kh, kw)
    int kw = idx % kernel_size;
    int kh = (idx / kernel_size) % kernel_size;
    int ic = (idx / (kernel_size * kernel_size)) % in_channels;
    int oc = idx / (kernel_size * kernel_size * in_channels);
    
    float sum = 0.0f;
    
    for (int n = 0; n < batch_size; ++n) {
        for (int oh = 0; oh < out_height; ++oh) {
            for (int ow = 0; ow < out_width; ++ow) {
                int ih = oh * stride - padding + kh;
                int iw = ow * stride - padding + kw;
                
                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                    int input_idx = ((n * in_channels + ic) * in_height + ih) * in_width + iw;
                    int grad_out_idx = ((n * out_channels + oc) * out_height + oh) * out_width + ow;
                    sum += input[input_idx] * grad_output[grad_out_idx];
                }
            }
        }
    }
    
    grad_weights[idx] = sum;
}

// Conv2D Backward Kernel - Bias Gradients
__global__ void conv2d_backward_bias_kernel(
    const float* grad_output, float* grad_bias,
    int batch_size, int out_channels, int out_height, int out_width
) {
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (oc >= out_channels) return;
    
    float sum = 0.0f;
    
    for (int n = 0; n < batch_size; ++n) {
        for (int oh = 0; oh < out_height; ++oh) {
            for (int ow = 0; ow < out_width; ++ow) {
                int idx = ((n * out_channels + oc) * out_height + oh) * out_width + ow;
                sum += grad_output[idx];
            }
        }
    }
    
    grad_bias[oc] = sum;
}

// Conv2D Backward Kernel - Input Gradients
__global__ void conv2d_backward_input_kernel(
    const float* weights, const float* grad_output,
    float* grad_input,
    int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int kernel_size, int stride, int padding,
    int out_height, int out_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_inputs = batch_size * in_channels * in_height * in_width;
    
    if (idx >= total_inputs) return;
    
    // Decode to (n, ic, ih, iw)
    int iw = idx % in_width;
    int ih = (idx / in_width) % in_height;
    int ic = (idx / (in_width * in_height)) % in_channels;
    int n = idx / (in_width * in_height * in_channels);
    
    float sum = 0.0f;
    
    for (int oc = 0; oc < out_channels; ++oc) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                // Find output position that this input contributes to
                int oh = (ih + padding - kh);
                int ow = (iw + padding - kw);
                
                if (oh % stride == 0 && ow % stride == 0) {
                    oh /= stride;
                    ow /= stride;
                    
                    if (oh >= 0 && oh < out_height && ow >= 0 && ow < out_width) {
                        int weight_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                        int grad_out_idx = ((n * out_channels + oc) * out_height + oh) * out_width + ow;
                        sum += weights[weight_idx] * grad_output[grad_out_idx];
                    }
                }
            }
        }
    }
    
    grad_input[idx] = sum;
}

// ReLU Forward Kernel
__global__ void relu_forward_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// ReLU Backward Kernel
__global__ void relu_backward_kernel(
    const float* input, const float* grad_output,
    float* grad_input, int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_input[idx] = (input[idx] > 0.0f) ? grad_output[idx] : 0.0f;
    }
}

// MaxPool2D Forward Kernel
__global__ void maxpool2d_forward_kernel(
    const float* input, float* output, int* mask,
    int batch_size, int channels, int in_height, int in_width,
    int pool_size, int stride, int out_height, int out_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels * out_height * out_width;
    
    if (idx >= total) return;
    
    // Decode to (n, c, oh, ow)
    int ow = idx % out_width;
    int oh = (idx / out_width) % out_height;
    int c = (idx / (out_width * out_height)) % channels;
    int n = idx / (out_width * out_height * channels);
    
    float max_val = -1e38f;
    int max_idx = 0;
    
    for (int ph = 0; ph < pool_size; ++ph) {
        for (int pw = 0; pw < pool_size; ++pw) {
            int ih = oh * stride + ph;
            int iw = ow * stride + pw;
            int input_idx = ((n * channels + c) * in_height + ih) * in_width + iw;
            
            if (input[input_idx] > max_val) {
                max_val = input[input_idx];
                max_idx = input_idx;
            }
        }
    }
    
    output[idx] = max_val;
    mask[idx] = max_idx;
}

// MaxPool2D Backward Kernel
__global__ void maxpool2d_backward_kernel(
    const float* grad_output, const int* mask, float* grad_input,
    int total_output, int total_input
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total_output) return;
    
    int max_idx = mask[idx];
    atomicAdd(&grad_input[max_idx], grad_output[idx]);
}

// Upsample2D Forward Kernel
__global__ void upsample2d_forward_kernel(
    const float* input, float* output,
    int batch_size, int channels, int in_height, int in_width,
    int scale_factor, int out_height, int out_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels * out_height * out_width;
    
    if (idx >= total) return;
    
    // Decode to (n, c, oh, ow)
    int ow = idx % out_width;
    int oh = (idx / out_width) % out_height;
    int c = (idx / (out_width * out_height)) % channels;
    int n = idx / (out_width * out_height * channels);
    
    int ih = oh / scale_factor;
    int iw = ow / scale_factor;
    
    int input_idx = ((n * channels + c) * in_height + ih) * in_width + iw;
    output[idx] = input[input_idx];
}

// Upsample2D Backward Kernel
__global__ void upsample2d_backward_kernel(
    const float* grad_output, float* grad_input,
    int batch_size, int channels, int in_height, int in_width,
    int scale_factor, int out_height, int out_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_input = batch_size * channels * in_height * in_width;
    
    if (idx >= total_input) return;
    
    // Decode to (n, c, ih, iw)
    int iw = idx % in_width;
    int ih = (idx / in_width) % in_height;
    int c = (idx / (in_width * in_height)) % channels;
    int n = idx / (in_width * in_height * channels);
    
    float sum = 0.0f;
    
    // Sum over all output positions that map to this input
    for (int sh = 0; sh < scale_factor; ++sh) {
        for (int sw = 0; sw < scale_factor; ++sw) {
            int oh = ih * scale_factor + sh;
            int ow = iw * scale_factor + sw;
            int out_idx = ((n * channels + c) * out_height + oh) * out_width + ow;
            sum += grad_output[out_idx];
        }
    }
    
    grad_input[idx] = sum;
}

// MSE Loss Forward Kernel (Reduction)
__global__ void mse_loss_kernel(
    const float* predicted, const float* target,
    float* partial_sums, int size
) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread computes sum of squared differences
    float sum = 0.0f;
    while (idx < size) {
        float diff = predicted[idx] - target[idx];
        sum += diff * diff;
        idx += blockDim.x * gridDim.x;
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

// MSE Loss Backward Kernel
__global__ void mse_loss_backward_kernel(
    const float* predicted, const float* target,
    float* grad_output, int size, float scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        grad_output[idx] = scale * (predicted[idx] - target[idx]);
    }
}

// SGD Update Kernel
__global__ void sgd_update_kernel(
    float* weights, const float* gradients,
    float learning_rate, int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        weights[idx] -= learning_rate * gradients[idx];
    }
}

// ============================================================================
// Host Functions
// ============================================================================

void conv2d_forward(
    const float* d_input, const float* d_weights, const float* d_bias,
    float* d_output,
    int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int kernel_size, int stride, int padding
) {
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    int total_outputs = batch_size * out_channels * out_height * out_width;
    
    int block_size = 256;
    int grid_size = (total_outputs + block_size - 1) / block_size;
    
    conv2d_forward_kernel<<<grid_size, block_size>>>(
        d_input, d_weights, d_bias, d_output,
        batch_size, in_channels, in_height, in_width,
        out_channels, kernel_size, stride, padding,
        out_height, out_width
    );
    
    CUDA_KERNEL_CHECK();
}

void conv2d_backward(
    const float* d_input, const float* d_weights, const float* d_grad_output,
    float* d_grad_input, float* d_grad_weights, float* d_grad_bias,
    int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int kernel_size, int stride, int padding
) {
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    int block_size = 256;
    
    // Zero gradients
    int total_inputs = batch_size * in_channels * in_height * in_width;
    cudaMemset(d_grad_input, 0, total_inputs * sizeof(float));
    
    // Compute weight gradients
    int total_weights = out_channels * in_channels * kernel_size * kernel_size;
    int grid_size_weights = (total_weights + block_size - 1) / block_size;
    conv2d_backward_weights_kernel<<<grid_size_weights, block_size>>>(
        d_input, d_grad_output, d_grad_weights, d_grad_bias,
        batch_size, in_channels, in_height, in_width,
        out_channels, kernel_size, stride, padding,
        out_height, out_width
    );
    
    // Compute bias gradients
    int grid_size_bias = (out_channels + block_size - 1) / block_size;
    conv2d_backward_bias_kernel<<<grid_size_bias, block_size>>>(
        d_grad_output, d_grad_bias,
        batch_size, out_channels, out_height, out_width
    );
    
    // Compute input gradients
    int grid_size_input = (total_inputs + block_size - 1) / block_size;
    conv2d_backward_input_kernel<<<grid_size_input, block_size>>>(
        d_weights, d_grad_output, d_grad_input,
        batch_size, in_channels, in_height, in_width,
        out_channels, kernel_size, stride, padding,
        out_height, out_width
    );
    
    CUDA_KERNEL_CHECK();
}

void relu_forward(const float* d_input, float* d_output, int size) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    relu_forward_kernel<<<grid_size, block_size>>>(d_input, d_output, size);
    CUDA_KERNEL_CHECK();
}

void relu_backward(const float* d_input, const float* d_grad_output,
                   float* d_grad_input, int size) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    relu_backward_kernel<<<grid_size, block_size>>>(
        d_input, d_grad_output, d_grad_input, size
    );
    CUDA_KERNEL_CHECK();
}

void maxpool2d_forward(
    const float* d_input, float* d_output, int* d_mask,
    int batch_size, int channels, int in_height, int in_width,
    int pool_size, int stride
) {
    int out_height = (in_height - pool_size) / stride + 1;
    int out_width = (in_width - pool_size) / stride + 1;
    int total = batch_size * channels * out_height * out_width;
    
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    
    maxpool2d_forward_kernel<<<grid_size, block_size>>>(
        d_input, d_output, d_mask,
        batch_size, channels, in_height, in_width,
        pool_size, stride, out_height, out_width
    );
    CUDA_KERNEL_CHECK();
}

void maxpool2d_backward(
    const float* d_grad_output, const int* d_mask, float* d_grad_input,
    int batch_size, int channels, int in_height, int in_width,
    int pool_size, int stride
) {
    int out_height = (in_height - pool_size) / stride + 1;
    int out_width = (in_width - pool_size) / stride + 1;
    int total_output = batch_size * channels * out_height * out_width;
    int total_input = batch_size * channels * in_height * in_width;
    
    // Zero gradient input
    cudaMemset(d_grad_input, 0, total_input * sizeof(float));
    
    int block_size = 256;
    int grid_size = (total_output + block_size - 1) / block_size;
    
    maxpool2d_backward_kernel<<<grid_size, block_size>>>(
        d_grad_output, d_mask, d_grad_input, total_output, total_input
    );
    CUDA_KERNEL_CHECK();
}

void upsample2d_forward(
    const float* d_input, float* d_output,
    int batch_size, int channels, int in_height, int in_width,
    int scale_factor
) {
    int out_height = in_height * scale_factor;
    int out_width = in_width * scale_factor;
    int total = batch_size * channels * out_height * out_width;
    
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    
    upsample2d_forward_kernel<<<grid_size, block_size>>>(
        d_input, d_output,
        batch_size, channels, in_height, in_width,
        scale_factor, out_height, out_width
    );
    CUDA_KERNEL_CHECK();
}

void upsample2d_backward(
    const float* d_grad_output, float* d_grad_input,
    int batch_size, int channels, int in_height, int in_width,
    int scale_factor
) {
    int out_height = in_height * scale_factor;
    int out_width = in_width * scale_factor;
    int total_input = batch_size * channels * in_height * in_width;
    
    int block_size = 256;
    int grid_size = (total_input + block_size - 1) / block_size;
    
    upsample2d_backward_kernel<<<grid_size, block_size>>>(
        d_grad_output, d_grad_input,
        batch_size, channels, in_height, in_width,
        scale_factor, out_height, out_width
    );
    CUDA_KERNEL_CHECK();
}

float mse_loss_forward(const float* d_predicted, const float* d_target, int size) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    grid_size = min(grid_size, 1024);  // Limit number of blocks
    
    // Use static buffer to avoid repeated allocation
    static float* d_partial_sums = nullptr;
    static float* h_partial_sums = nullptr;
    static int allocated_grid_size = 0;
    
    if (d_partial_sums == nullptr || grid_size > allocated_grid_size) {
        if (d_partial_sums != nullptr) {
            cudaFree(d_partial_sums);
            delete[] h_partial_sums;
        }
        cudaMalloc(&d_partial_sums, grid_size * sizeof(float));
        h_partial_sums = new float[grid_size];
        allocated_grid_size = grid_size;
    }
    
    mse_loss_kernel<<<grid_size, block_size, block_size * sizeof(float)>>>(
        d_predicted, d_target, d_partial_sums, size
    );
    
    // Copy partial sums to host and reduce
    cudaMemcpy(h_partial_sums, d_partial_sums, grid_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    float total_sum = 0.0f;
    for (int i = 0; i < grid_size; ++i) {
        total_sum += h_partial_sums[i];
    }
    
    return total_sum / size;
}

void mse_loss_backward(const float* d_predicted, const float* d_target,
                       float* d_grad_output, int size) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    float scale = 2.0f / size;
    
    mse_loss_backward_kernel<<<grid_size, block_size>>>(
        d_predicted, d_target, d_grad_output, size, scale
    );
    CUDA_KERNEL_CHECK();
}

void sgd_update(float* d_weights, const float* d_gradients,
                float learning_rate, int size) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    sgd_update_kernel<<<grid_size, block_size>>>(
        d_weights, d_gradients, learning_rate, size
    );
    CUDA_KERNEL_CHECK();
}

} // namespace gpu


