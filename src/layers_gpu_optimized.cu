#include "layers.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace gpu_optimized {

// ============================================================================
// Optimized CUDA Kernels
// ============================================================================

// Tile dimensions for shared memory tiling
#define TILE_SIZE 16
#define TILE_SIZE_LARGE 32  // For larger feature maps
#define KERNEL_RADIUS 1  // For 3x3 kernel
#define WARP_SIZE 32

// ============================================================================
// Warp-level reduction utilities
// ============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float shared[32];  // One per warp
    
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    val = warp_reduce_sum(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    // Only first warp does final reduction
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    
    return val;
}

// ============================================================================
// Version 1: Optimized Convolution with Weight Caching
// For small images (32x32), shared memory tiling has too much overhead.
// Instead, we cache weights and use register blocking.
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
    // (weight caching via shared memory was considered but not used here)
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * out_channels * out_height * out_width;
    
    if (idx >= total_outputs) return;
    
    // Decode linear index to (n, oc, oh, ow)
    int ow = idx % out_width;
    int oh = (idx / out_width) % out_height;
    int oc = (idx / (out_width * out_height)) % out_channels;
    int n = idx / (out_width * out_height * out_channels);
    
    float sum = 0.0f;
    
    // Convolution with loop unrolling
    for (int ic = 0; ic < in_channels; ++ic) {
        const float* weight_ptr = weights + (oc * in_channels + ic) * kernel_size * kernel_size;
        const float* input_ptr = input + (n * in_channels + ic) * in_height * in_width;
        
        #pragma unroll
        for (int kh = 0; kh < 3; ++kh) {
            int ih = oh * stride - padding + kh;
            if (ih < 0 || ih >= in_height) continue;
            
            #pragma unroll
            for (int kw = 0; kw < 3; ++kw) {
                int iw = ow * stride - padding + kw;
                
                if (iw >= 0 && iw < in_width) {
                    sum += input_ptr[ih * in_width + iw] * weight_ptr[kh * kernel_size + kw];
                }
            }
        }
    }
    
    // Add bias
    output[idx] = sum + bias[oc];
}

// ============================================================================
// Version 2: Fused Conv2D + ReLU + Bias (efficient for small images)
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * out_channels * out_height * out_width;
    
    if (idx >= total_outputs) return;
    
    // Decode linear index to (n, oc, oh, ow)
    int ow = idx % out_width;
    int oh = (idx / out_width) % out_height;
    int oc = (idx / (out_width * out_height)) % out_channels;
    int n = idx / (out_width * out_height * out_channels);
    
    float sum = 0.0f;
    
    // Convolution with loop unrolling
    for (int ic = 0; ic < in_channels; ++ic) {
        const float* weight_ptr = weights + (oc * in_channels + ic) * kernel_size * kernel_size;
        const float* input_ptr = input + (n * in_channels + ic) * in_height * in_width;
        
        #pragma unroll
        for (int kh = 0; kh < 3; ++kh) {
            int ih = oh * stride - padding + kh;
            if (ih < 0 || ih >= in_height) continue;
            
            #pragma unroll
            for (int kw = 0; kw < 3; ++kw) {
                int iw = ow * stride - padding + kw;
                
                if (iw >= 0 && iw < in_width) {
                    sum += input_ptr[ih * in_width + iw] * weight_ptr[kh * kernel_size + kw];
                }
            }
        }
    }
    
    // Fused: Add bias + ReLU
    float val = sum + bias[oc];
    output[idx] = fmaxf(0.0f, val);
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

// Optimized bias gradient - simple and fast for small spatial sizes
__global__ void conv2d_backward_bias_optimized_kernel(
    const float* __restrict__ grad_output,
    float* __restrict__ grad_bias,
    int batch_size, int out_channels, int out_height, int out_width
) {
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc >= out_channels) return;
    
    int spatial_size = out_height * out_width;
    float sum = 0.0f;
    
    // Simple accumulation - efficient for small spatial sizes
    for (int n = 0; n < batch_size; ++n) {
        const float* grad_ptr = grad_output + (n * out_channels + oc) * spatial_size;
        #pragma unroll 4
        for (int i = 0; i < spatial_size; ++i) {
            sum += grad_ptr[i];
        }
    }
    
    grad_bias[oc] = sum;
}

// Optimized weight gradients - parallelize over all weight elements
__global__ void conv2d_backward_weights_optimized_kernel(
    const float* __restrict__ input,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_weights,
    int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int kernel_size, int stride, int padding,
    int out_height, int out_width
) {
    // Each thread handles one weight element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_weights = out_channels * in_channels * kernel_size * kernel_size;
    
    if (idx >= total_weights) return;
    
    int kw = idx % kernel_size;
    int kh = (idx / kernel_size) % kernel_size;
    int ic = (idx / (kernel_size * kernel_size)) % in_channels;
    int oc = idx / (kernel_size * kernel_size * in_channels);
    
    float sum = 0.0f;
    int spatial_size = out_height * out_width;
    
    for (int n = 0; n < batch_size; ++n) {
        const float* input_base = input + (n * in_channels + ic) * in_height * in_width;
        const float* grad_base = grad_output + (n * out_channels + oc) * spatial_size;
        
        for (int oh = 0; oh < out_height; ++oh) {
            int ih = oh * stride - padding + kh;
            if (ih < 0 || ih >= in_height) continue;
            
            #pragma unroll 4
            for (int ow = 0; ow < out_width; ++ow) {
                int iw = ow * stride - padding + kw;
                
                if (iw >= 0 && iw < in_width) {
                    sum += input_base[ih * in_width + iw] * grad_base[oh * out_width + ow];
                }
            }
        }
    }
    
    grad_weights[idx] = sum;
}

// Optimized input gradients - simplified for stride=1 (most common case)
__global__ void conv2d_backward_input_optimized_kernel(
    const float* __restrict__ weights,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int kernel_size, int stride, int padding,
    int out_height, int out_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_inputs = batch_size * in_channels * in_height * in_width;
    
    if (idx >= total_inputs) return;
    
    int iw = idx % in_width;
    int ih = (idx / in_width) % in_height;
    int ic = (idx / (in_width * in_height)) % in_channels;
    int n = idx / (in_width * in_height * in_channels);
    
    float sum = 0.0f;
    int spatial_out = out_height * out_width;
    
    // Optimized for stride=1 case (most common in autoencoders)
    if (stride == 1) {
        for (int oc = 0; oc < out_channels; ++oc) {
            const float* weight_base = weights + (oc * in_channels + ic) * kernel_size * kernel_size;
            const float* grad_base = grad_output + (n * out_channels + oc) * spatial_out;
            
            #pragma unroll
            for (int kh = 0; kh < 3; ++kh) {
                int oh = ih + padding - kh;
                if (oh < 0 || oh >= out_height) continue;
                
                #pragma unroll
                for (int kw = 0; kw < 3; ++kw) {
                    int ow = iw + padding - kw;
                    if (ow >= 0 && ow < out_width) {
                        sum += weight_base[kh * kernel_size + kw] * grad_base[oh * out_width + ow];
                    }
                }
            }
        }
    } else {
        // General case for stride > 1
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int oh = (ih + padding - kh);
                    int ow = (iw + padding - kw);
                    
                    if (oh % stride == 0 && ow % stride == 0) {
                        oh /= stride;
                        ow /= stride;
                        if (oh >= 0 && oh < out_height && ow >= 0 && ow < out_width) {
                            int weight_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                            sum += weights[weight_idx] * grad_output[(n * out_channels + oc) * spatial_out + oh * out_width + ow];
                        }
                    }
                }
            }
        }
    }
    
    grad_input[idx] = sum;
}

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
    
    int total_outputs = batch_size * out_channels * out_height * out_width;
    int block_size = 256;
    int grid_size = (total_outputs + block_size - 1) / block_size;
    
    conv2d_forward_tiled_kernel<<<grid_size, block_size>>>(
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
    
    int total_outputs = batch_size * out_channels * out_height * out_width;
    int block_size = 256;
    int grid_size = (total_outputs + block_size - 1) / block_size;
    
    conv2d_relu_forward_kernel<<<grid_size, block_size>>>(
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
    
    // Optimized bias gradients - simple parallel over channels
    int bias_block = 256;
    int bias_grid = (out_channels + bias_block - 1) / bias_block;
    conv2d_backward_bias_optimized_kernel<<<bias_grid, bias_block>>>(
        d_grad_output, d_grad_bias,
        batch_size, out_channels, out_height, out_width
    );
    CUDA_KERNEL_CHECK();
    
    // Optimized weight gradients - one thread per weight
    int total_weights = out_channels * in_channels * kernel_size * kernel_size;
    int weight_block = 256;
    int weight_grid = (total_weights + weight_block - 1) / weight_block;
    conv2d_backward_weights_optimized_kernel<<<weight_grid, weight_block>>>(
        d_input, d_grad_output, d_grad_weights,
        batch_size, in_channels, in_height, in_width,
        out_channels, kernel_size, stride, padding,
        out_height, out_width
    );
    CUDA_KERNEL_CHECK();
    
    // Optimized input gradients
    int total_inputs = batch_size * in_channels * in_height * in_width;
    int block_size = 256;
    int grid_size = (total_inputs + block_size - 1) / block_size;
    
    conv2d_backward_input_optimized_kernel<<<grid_size, block_size>>>(
        d_weights, d_grad_output, d_grad_input,
        batch_size, in_channels, in_height, in_width,
        out_channels, kernel_size, stride, padding,
        out_height, out_width
    );
    CUDA_KERNEL_CHECK();
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

