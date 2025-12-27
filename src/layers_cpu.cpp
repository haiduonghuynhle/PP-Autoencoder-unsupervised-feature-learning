#include "layers.h"
#include <cstring>
#include <cmath>
#include <random>
#include <algorithm>

namespace cpu {

// ============================================================================
// Convolution 2D
// ============================================================================

void conv2d_forward(
    const float* input, const float* weights, const float* bias,
    float* output,
    int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int kernel_size, int stride, int padding
) {
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    // Zero output first
    int output_size = batch_size * out_channels * out_height * out_width;
    memset(output, 0, output_size * sizeof(float));
    
    // For each image in batch
    for (int n = 0; n < batch_size; ++n) {
        // For each output channel
        for (int oc = 0; oc < out_channels; ++oc) {
            // For each output position
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
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
                    
                    int output_idx = ((n * out_channels + oc) * out_height + oh) * out_width + ow;
                    output[output_idx] = sum;
                }
            }
        }
    }
}

void conv2d_backward(
    const float* input, const float* weights, const float* grad_output,
    float* grad_input, float* grad_weights, float* grad_bias,
    int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int kernel_size, int stride, int padding
) {
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    // Zero gradients
    int input_size = batch_size * in_channels * in_height * in_width;
    int weight_size = out_channels * in_channels * kernel_size * kernel_size;
    
    memset(grad_input, 0, input_size * sizeof(float));
    memset(grad_weights, 0, weight_size * sizeof(float));
    memset(grad_bias, 0, out_channels * sizeof(float));
    
    // Compute gradients
    for (int n = 0; n < batch_size; ++n) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    int grad_out_idx = ((n * out_channels + oc) * out_height + oh) * out_width + ow;
                    float grad_out_val = grad_output[grad_out_idx];
                    
                    // Gradient w.r.t. bias
                    grad_bias[oc] += grad_out_val;
                    
                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int kh = 0; kh < kernel_size; ++kh) {
                            for (int kw = 0; kw < kernel_size; ++kw) {
                                int ih = oh * stride - padding + kh;
                                int iw = ow * stride - padding + kw;
                                
                                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                    int input_idx = ((n * in_channels + ic) * in_height + ih) * in_width + iw;
                                    int weight_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                                    
                                    // Gradient w.r.t. weights
                                    grad_weights[weight_idx] += input[input_idx] * grad_out_val;
                                    
                                    // Gradient w.r.t. input
                                    grad_input[input_idx] += weights[weight_idx] * grad_out_val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// ReLU Activation
// ============================================================================

void relu_forward(const float* input, float* output, int size) {
    for (int i = 0; i < size; ++i) {
        output[i] = std::max(0.0f, input[i]);
    }
}

void relu_backward(const float* input, const float* grad_output, 
                   float* grad_input, int size) {
    for (int i = 0; i < size; ++i) {
        grad_input[i] = (input[i] > 0.0f) ? grad_output[i] : 0.0f;
    }
}

// ============================================================================
// Max Pooling 2D
// ============================================================================

void maxpool2d_forward(
    const float* input, float* output, int* mask,
    int batch_size, int channels, int in_height, int in_width,
    int pool_size, int stride
) {
    int out_height = (in_height - pool_size) / stride + 1;
    int out_width = (in_width - pool_size) / stride + 1;
    
    for (int n = 0; n < batch_size; ++n) {
        for (int c = 0; c < channels; ++c) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
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
                    
                    int output_idx = ((n * channels + c) * out_height + oh) * out_width + ow;
                    output[output_idx] = max_val;
                    mask[output_idx] = max_idx;
                }
            }
        }
    }
}

void maxpool2d_backward(
    const float* grad_output, const int* mask, float* grad_input,
    int batch_size, int channels, int in_height, int in_width,
    int pool_size, int stride
) {
    int out_height = (in_height - pool_size) / stride + 1;
    int out_width = (in_width - pool_size) / stride + 1;
    
    // Zero gradient input
    int input_size = batch_size * channels * in_height * in_width;
    memset(grad_input, 0, input_size * sizeof(float));
    
    // Propagate gradients to max positions
    for (int n = 0; n < batch_size; ++n) {
        for (int c = 0; c < channels; ++c) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    int output_idx = ((n * channels + c) * out_height + oh) * out_width + ow;
                    int max_idx = mask[output_idx];
                    grad_input[max_idx] += grad_output[output_idx];
                }
            }
        }
    }
}

// ============================================================================
// Upsampling 2D (Nearest Neighbor)
// ============================================================================

void upsample2d_forward(
    const float* input, float* output,
    int batch_size, int channels, int in_height, int in_width,
    int scale_factor
) {
    int out_height = in_height * scale_factor;
    int out_width = in_width * scale_factor;
    
    for (int n = 0; n < batch_size; ++n) {
        for (int c = 0; c < channels; ++c) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    int ih = oh / scale_factor;
                    int iw = ow / scale_factor;
                    
                    int input_idx = ((n * channels + c) * in_height + ih) * in_width + iw;
                    int output_idx = ((n * channels + c) * out_height + oh) * out_width + ow;
                    
                    output[output_idx] = input[input_idx];
                }
            }
        }
    }
}

void upsample2d_backward(
    const float* grad_output, float* grad_input,
    int batch_size, int channels, int in_height, int in_width,
    int scale_factor
) {
    int out_height = in_height * scale_factor;
    int out_width = in_width * scale_factor;
    
    // Zero gradient input
    int input_size = batch_size * channels * in_height * in_width;
    memset(grad_input, 0, input_size * sizeof(float));
    
    // Accumulate gradients (sum over the upsampled region)
    for (int n = 0; n < batch_size; ++n) {
        for (int c = 0; c < channels; ++c) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    int ih = oh / scale_factor;
                    int iw = ow / scale_factor;
                    
                    int input_idx = ((n * channels + c) * in_height + ih) * in_width + iw;
                    int output_idx = ((n * channels + c) * out_height + oh) * out_width + ow;
                    
                    grad_input[input_idx] += grad_output[output_idx];
                }
            }
        }
    }
}

// ============================================================================
// MSE Loss
// ============================================================================

float mse_loss_forward(const float* predicted, const float* target, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        float diff = predicted[i] - target[i];
        sum += diff * diff;
    }
    return sum / size;
}

void mse_loss_backward(const float* predicted, const float* target,
                       float* grad_output, int size) {
    float scale = 2.0f / size;
    for (int i = 0; i < size; ++i) {
        grad_output[i] = scale * (predicted[i] - target[i]);
    }
}

// ============================================================================
// SGD Update
// ============================================================================

void sgd_update(float* weights, const float* gradients, 
                float learning_rate, int size) {
    for (int i = 0; i < size; ++i) {
        weights[i] -= learning_rate * gradients[i];
    }
}

// ============================================================================
// Weight Initialization
// ============================================================================

void initialize_weights_he(float* weights, int fan_in, int size, unsigned int seed) {
    std::mt19937 gen(seed);
    float std_dev = std::sqrt(2.0f / fan_in);
    std::normal_distribution<float> dist(0.0f, std_dev);
    
    for (int i = 0; i < size; ++i) {
        weights[i] = dist(gen);
    }
}

void initialize_bias_zero(float* bias, int size) {
    memset(bias, 0, size * sizeof(float));
}

} // namespace cpu


