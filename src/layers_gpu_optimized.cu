#include "layers.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace gpu_optimized {

// ============================================================================
// Optimized CUDA Kernels
// ============================================================================

// Tile dimensions for shared memory tiling
#define TILE_SIZE 16
#define KERNEL_RADIUS 1  // For 3x3 kernel

// ============================================================================
// Version 1: Shared Memory Tiling for Convolution
// ============================================================================

__global__ void conv2d_forward_tiled_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weights, 
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int kernel_size, int stride, int padding,
    int out_height, int out_width
) {
    // Shared memory for input tile (with halo region)
    __shared__ float s_input[TILE_SIZE + 2][TILE_SIZE + 2];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;  // batch * out_channels
    
    int n = bz / out_channels;
    int oc = bz % out_channels;
    
    int oh = by * TILE_SIZE + ty;
    int ow = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Loop over input channels
    for (int ic = 0; ic < in_channels; ++ic) {
        // Load input tile into shared memory
        int ih_base = by * TILE_SIZE - padding;
        int iw_base = bx * TILE_SIZE - padding;
        
        // Each thread loads one element of the base tile
        int ih = ih_base + ty;
        int iw = iw_base + tx;
        
        // Load with boundary check
        if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
            s_input[ty][tx] = input[((n * in_channels + ic) * in_height + ih) * in_width + iw];
        } else {
            s_input[ty][tx] = 0.0f;
        }
        
        // Load halo regions (right and bottom borders)
        if (tx < 2) {
            int iw_halo = iw_base + TILE_SIZE + tx;
            if (ih >= 0 && ih < in_height && iw_halo >= 0 && iw_halo < in_width) {
                s_input[ty][TILE_SIZE + tx] = input[((n * in_channels + ic) * in_height + ih) * in_width + iw_halo];
            } else {
                s_input[ty][TILE_SIZE + tx] = 0.0f;
            }
        }
        
        if (ty < 2) {
            int ih_halo = ih_base + TILE_SIZE + ty;
            if (ih_halo >= 0 && ih_halo < in_height && iw >= 0 && iw < in_width) {
                s_input[TILE_SIZE + ty][tx] = input[((n * in_channels + ic) * in_height + ih_halo) * in_width + iw];
            } else {
                s_input[TILE_SIZE + ty][tx] = 0.0f;
            }
        }
        
        // Corner elements
        if (tx < 2 && ty < 2) {
            int ih_halo = ih_base + TILE_SIZE + ty;
            int iw_halo = iw_base + TILE_SIZE + tx;
            if (ih_halo >= 0 && ih_halo < in_height && iw_halo >= 0 && iw_halo < in_width) {
                s_input[TILE_SIZE + ty][TILE_SIZE + tx] = input[((n * in_channels + ic) * in_height + ih_halo) * in_width + iw_halo];
            } else {
                s_input[TILE_SIZE + ty][TILE_SIZE + tx] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // Compute convolution using shared memory
        if (oh < out_height && ow < out_width) {
            #pragma unroll
            for (int kh = 0; kh < 3; ++kh) {
                #pragma unroll
                for (int kw = 0; kw < 3; ++kw) {
                    int sy = ty + kh;
                    int sx = tx + kw;
                    int weight_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                    sum += s_input[sy][sx] * weights[weight_idx];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write output with bias
    if (oh < out_height && ow < out_width) {
        int output_idx = ((n * out_channels + oc) * out_height + oh) * out_width + ow;
        output[output_idx] = sum + bias[oc];
    }
}

// ============================================================================
// Version 2: Fused Conv2D + ReLU + Bias
// ============================================================================

__global__ void conv2d_relu_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int kernel_size, int stride, int padding,
    int out_height, int out_width
) {
    __shared__ float s_input[TILE_SIZE + 2][TILE_SIZE + 2];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;
    
    int n = bz / out_channels;
    int oc = bz % out_channels;
    
    int oh = by * TILE_SIZE + ty;
    int ow = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int ic = 0; ic < in_channels; ++ic) {
        int ih_base = by * TILE_SIZE - padding;
        int iw_base = bx * TILE_SIZE - padding;
        
        int ih = ih_base + ty;
        int iw = iw_base + tx;
        
        // Load main tile
        if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
            s_input[ty][tx] = input[((n * in_channels + ic) * in_height + ih) * in_width + iw];
        } else {
            s_input[ty][tx] = 0.0f;
        }
        
        // Load halo - right
        if (tx < 2) {
            int iw_halo = iw_base + TILE_SIZE + tx;
            if (ih >= 0 && ih < in_height && iw_halo >= 0 && iw_halo < in_width) {
                s_input[ty][TILE_SIZE + tx] = input[((n * in_channels + ic) * in_height + ih) * in_width + iw_halo];
            } else {
                s_input[ty][TILE_SIZE + tx] = 0.0f;
            }
        }
        
        // Load halo - bottom
        if (ty < 2) {
            int ih_halo = ih_base + TILE_SIZE + ty;
            if (ih_halo >= 0 && ih_halo < in_height && iw >= 0 && iw < in_width) {
                s_input[TILE_SIZE + ty][tx] = input[((n * in_channels + ic) * in_height + ih_halo) * in_width + iw];
            } else {
                s_input[TILE_SIZE + ty][tx] = 0.0f;
            }
        }
        
        // Load halo - corner
        if (tx < 2 && ty < 2) {
            int ih_halo = ih_base + TILE_SIZE + ty;
            int iw_halo = iw_base + TILE_SIZE + tx;
            if (ih_halo >= 0 && ih_halo < in_height && iw_halo >= 0 && iw_halo < in_width) {
                s_input[TILE_SIZE + ty][TILE_SIZE + tx] = input[((n * in_channels + ic) * in_height + ih_halo) * in_width + iw_halo];
            } else {
                s_input[TILE_SIZE + ty][TILE_SIZE + tx] = 0.0f;
            }
        }
        
        __syncthreads();
        
        if (oh < out_height && ow < out_width) {
            #pragma unroll
            for (int kh = 0; kh < 3; ++kh) {
                #pragma unroll
                for (int kw = 0; kw < 3; ++kw) {
                    int weight_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                    sum += s_input[ty + kh][tx + kw] * weights[weight_idx];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Fused: Add bias + ReLU
    if (oh < out_height && ow < out_width) {
        int output_idx = ((n * out_channels + oc) * out_height + oh) * out_width + ow;
        float val = sum + bias[oc];
        output[output_idx] = fmaxf(0.0f, val);  // ReLU fused
    }
}

// ============================================================================
// Vectorized ReLU using float4
// ============================================================================

__global__ void relu_forward_vectorized_kernel(
    const float4* __restrict__ input,
    float4* __restrict__ output,
    int size4
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size4) {
        float4 in = input[idx];
        float4 out;
        out.x = fmaxf(0.0f, in.x);
        out.y = fmaxf(0.0f, in.y);
        out.z = fmaxf(0.0f, in.z);
        out.w = fmaxf(0.0f, in.w);
        output[idx] = out;
    }
}

__global__ void relu_backward_vectorized_kernel(
    const float4* __restrict__ input,
    const float4* __restrict__ grad_output,
    float4* __restrict__ grad_input,
    int size4
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size4) {
        float4 in = input[idx];
        float4 grad_out = grad_output[idx];
        float4 grad_in;
        grad_in.x = (in.x > 0.0f) ? grad_out.x : 0.0f;
        grad_in.y = (in.y > 0.0f) ? grad_out.y : 0.0f;
        grad_in.z = (in.z > 0.0f) ? grad_out.z : 0.0f;
        grad_in.w = (in.w > 0.0f) ? grad_out.w : 0.0f;
        grad_input[idx] = grad_in;
    }
}

// ============================================================================
// Optimized Upsample with Coalesced Memory Access
// ============================================================================

__global__ void upsample2d_forward_coalesced_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels, int in_height, int in_width,
    int out_height, int out_width
) {
    // Process 4 output elements at once for better coalescing
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels * out_height * out_width;
    
    if (idx >= total) return;
    
    int ow = idx % out_width;
    int oh = (idx / out_width) % out_height;
    int c = (idx / (out_width * out_height)) % channels;
    int n = idx / (out_width * out_height * channels);
    
    // Nearest neighbor: divide by 2
    int ih = oh >> 1;
    int iw = ow >> 1;
    
    int input_idx = ((n * channels + c) * in_height + ih) * in_width + iw;
    output[idx] = input[input_idx];
}

// ============================================================================
// Vectorized SGD Update
// ============================================================================

__global__ void sgd_update_vectorized_kernel(
    float4* __restrict__ weights,
    const float4* __restrict__ gradients,
    float learning_rate,
    int size4
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size4) {
        float4 w = weights[idx];
        float4 g = gradients[idx];
        w.x -= learning_rate * g.x;
        w.y -= learning_rate * g.y;
        w.z -= learning_rate * g.z;
        w.w -= learning_rate * g.w;
        weights[idx] = w;
    }
}

// Handle remaining elements (non-multiple of 4)
__global__ void sgd_update_remainder_kernel(
    float* weights,
    const float* gradients,
    float learning_rate,
    int start_idx,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + start_idx;
    
    if (idx < size) {
        weights[idx] -= learning_rate * gradients[idx];
    }
}

// ============================================================================
// Conv2D Backward with Shared Memory (Optimized)
// ============================================================================

__global__ void conv2d_backward_weights_tiled_kernel(
    const float* __restrict__ input,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_weights,
    int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int kernel_size, int stride, int padding,
    int out_height, int out_width
) {
    // Each block computes gradients for one (oc, ic, kh, kw) combination
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_weights = out_channels * in_channels * kernel_size * kernel_size;
    
    if (idx >= total_weights) return;
    
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

// ============================================================================
// Host Function Wrappers
// ============================================================================

void conv2d_forward_tiled(
    const float* d_input, const float* d_weights, const float* d_bias,
    float* d_output,
    int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int kernel_size, int stride, int padding
) {
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(
        (out_width + TILE_SIZE - 1) / TILE_SIZE,
        (out_height + TILE_SIZE - 1) / TILE_SIZE,
        batch_size * out_channels
    );
    
    conv2d_forward_tiled_kernel<<<grid, block>>>(
        d_input, d_weights, d_bias, d_output,
        batch_size, in_channels, in_height, in_width,
        out_channels, kernel_size, stride, padding,
        out_height, out_width
    );
    
    CUDA_KERNEL_CHECK();
}

void conv2d_relu_forward(
    const float* d_input, const float* d_weights, const float* d_bias,
    float* d_output,
    int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int kernel_size, int stride, int padding
) {
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(
        (out_width + TILE_SIZE - 1) / TILE_SIZE,
        (out_height + TILE_SIZE - 1) / TILE_SIZE,
        batch_size * out_channels
    );
    
    conv2d_relu_forward_kernel<<<grid, block>>>(
        d_input, d_weights, d_bias, d_output,
        batch_size, in_channels, in_height, in_width,
        out_channels, kernel_size, stride, padding,
        out_height, out_width
    );
    
    CUDA_KERNEL_CHECK();
}

void conv2d_backward_tiled(
    const float* d_input, const float* d_weights, const float* d_grad_output,
    float* d_grad_input, float* d_grad_weights, float* d_grad_bias,
    int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int kernel_size, int stride, int padding
) {
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    // Use the naive backward for now (optimization of backward pass is complex)
    gpu::conv2d_backward(d_input, d_weights, d_grad_output,
                         d_grad_input, d_grad_weights, d_grad_bias,
                         batch_size, in_channels, in_height, in_width,
                         out_channels, kernel_size, stride, padding);
}

void relu_forward_vectorized(const float* d_input, float* d_output, int size) {
    int size4 = size / 4;
    int remainder = size % 4;
    
    if (size4 > 0) {
        int block_size = 256;
        int grid_size = (size4 + block_size - 1) / block_size;
        
        relu_forward_vectorized_kernel<<<grid_size, block_size>>>(
            reinterpret_cast<const float4*>(d_input),
            reinterpret_cast<float4*>(d_output),
            size4
        );
    }
    
    // Handle remainder with regular kernel
    if (remainder > 0) {
        int start = size4 * 4;
        gpu::relu_forward(d_input + start, d_output + start, remainder);
    }
    
    CUDA_KERNEL_CHECK();
}

void relu_backward_vectorized(const float* d_input, const float* d_grad_output,
                              float* d_grad_input, int size) {
    int size4 = size / 4;
    int remainder = size % 4;
    
    if (size4 > 0) {
        int block_size = 256;
        int grid_size = (size4 + block_size - 1) / block_size;
        
        relu_backward_vectorized_kernel<<<grid_size, block_size>>>(
            reinterpret_cast<const float4*>(d_input),
            reinterpret_cast<const float4*>(d_grad_output),
            reinterpret_cast<float4*>(d_grad_input),
            size4
        );
    }
    
    if (remainder > 0) {
        int start = size4 * 4;
        gpu::relu_backward(d_input + start, d_grad_output + start, 
                           d_grad_input + start, remainder);
    }
    
    CUDA_KERNEL_CHECK();
}

void maxpool2d_forward_fused(
    const float* d_input, float* d_output, int* d_mask,
    int batch_size, int channels, int in_height, int in_width,
    int pool_size, int stride
) {
    // Use the naive version (pooling is already efficient)
    gpu::maxpool2d_forward(d_input, d_output, d_mask,
                           batch_size, channels, in_height, in_width,
                           pool_size, stride);
}

void upsample2d_forward_coalesced(
    const float* d_input, float* d_output,
    int batch_size, int channels, int in_height, int in_width,
    int scale_factor
) {
    int out_height = in_height * scale_factor;
    int out_width = in_width * scale_factor;
    int total = batch_size * channels * out_height * out_width;
    
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    
    upsample2d_forward_coalesced_kernel<<<grid_size, block_size>>>(
        d_input, d_output,
        batch_size, channels, in_height, in_width,
        out_height, out_width
    );
    
    CUDA_KERNEL_CHECK();
}

void sgd_update_vectorized(float* d_weights, const float* d_gradients,
                           float learning_rate, int size) {
    int size4 = size / 4;
    int remainder = size % 4;
    
    if (size4 > 0) {
        int block_size = 256;
        int grid_size = (size4 + block_size - 1) / block_size;
        
        sgd_update_vectorized_kernel<<<grid_size, block_size>>>(
            reinterpret_cast<float4*>(d_weights),
            reinterpret_cast<const float4*>(d_gradients),
            learning_rate,
            size4
        );
    }
    
    if (remainder > 0) {
        int start = size4 * 4;
        int block_size = 256;
        int grid_size = (remainder + block_size - 1) / block_size;
        
        sgd_update_remainder_kernel<<<grid_size, block_size>>>(
            d_weights, d_gradients, learning_rate, start, size
        );
    }
    
    CUDA_KERNEL_CHECK();
}

} // namespace gpu_optimized

