#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void add_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data) {
    
    for (int i = 0; i < tensor1->size; i++) {
        result_data[i] = tensor1->data[i] + tensor2->data[i];
    }
}

void add_broadcasted_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data, int* broadcasted_shape, int broadcasted_size) {
    int max_ndim = tensor1->ndim > tensor2->ndim ? tensor1->ndim : tensor2->ndim;

    // Calculate strides for broadcasting
    int* strides1 = (int*)malloc(max_ndim * sizeof(int));
    int* strides2 = (int*)malloc(max_ndim * sizeof(int));
    if (strides1 == NULL || strides2 == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    int stride1 = 1, stride2 = 1;
    for (int i = max_ndim - 1; i >= 0; i--) {
        int dim1 = i < tensor1->ndim ? tensor1->shape[tensor1->ndim - max_ndim + i] : 1;
        int dim2 = i < tensor2->ndim ? tensor2->shape[tensor2->ndim - max_ndim + i] : 1;
        strides1[i] = dim1 == broadcasted_shape[i] ? stride1 : 0;
        strides2[i] = dim2 == broadcasted_shape[i] ? stride2 : 0;
        stride1 *= (dim1 == broadcasted_shape[i]) ? dim1 : 1;
        stride2 *= (dim2 == broadcasted_shape[i]) ? dim2 : 1;
    }

    // Perform element-wise addition with broadcasting
    for (int i = 0; i < broadcasted_size; i++) {
        int index1 = 0, index2 = 0;
        int linear_index = i;
        for (int j = max_ndim - 1; j >= 0; j--) {
            int pos = linear_index % broadcasted_shape[j];
            linear_index /= broadcasted_shape[j];
            if (strides1[j] != 0) index1 += pos * strides1[j];
            if (strides2[j] != 0) index2 += pos * strides2[j];
        }
        result_data[i] = tensor1->data[index1] + tensor2->data[index2];
    }

    // Free strides
    free(strides1);
    free(strides2);
}


void sub_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data) {
    
    for (int i = 0; i < tensor1->size; i++) {
        result_data[i] = tensor1->data[i] - tensor2->data[i];
    }
}

void sub_broadcasted_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data, int* broadcasted_shape, int broadcasted_size) {
    int max_ndim = tensor1->ndim > tensor2->ndim ? tensor1->ndim : tensor2->ndim;

    // Calculate strides for broadcasting
    int* strides1 = (int*)malloc(max_ndim * sizeof(int));
    int* strides2 = (int*)malloc(max_ndim * sizeof(int));
    if (strides1 == NULL || strides2 == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    int stride1 = 1, stride2 = 1;
    for (int i = max_ndim - 1; i >= 0; i--) {
        int dim1 = i < tensor1->ndim ? tensor1->shape[tensor1->ndim - max_ndim + i] : 1;
        int dim2 = i < tensor2->ndim ? tensor2->shape[tensor2->ndim - max_ndim + i] : 1;
        strides1[i] = dim1 == broadcasted_shape[i] ? stride1 : 0;
        strides2[i] = dim2 == broadcasted_shape[i] ? stride2 : 0;
        stride1 *= (dim1 == broadcasted_shape[i]) ? dim1 : 1;
        stride2 *= (dim2 == broadcasted_shape[i]) ? dim2 : 1;
    }

    // Perform element-wise addition with broadcasting
    for (int i = 0; i < broadcasted_size; i++) {
        int index1 = 0, index2 = 0;
        int linear_index = i;
        for (int j = max_ndim - 1; j >= 0; j--) {
            int pos = linear_index % broadcasted_shape[j];
            linear_index /= broadcasted_shape[j];
            if (strides1[j] != 0) index1 += pos * strides1[j];
            if (strides2[j] != 0) index2 += pos * strides2[j];
        }
        result_data[i] = tensor1->data[index1] - tensor2->data[index2];
    }

    // Free strides
    free(strides1);
    free(strides2);
}

void elementwise_mul_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data) {
    
    for (int i = 0; i < tensor1->size; i++) {
        result_data[i] = tensor1->data[i] * tensor2->data[i];
    }
}

void scalar_mul_tensor_cpu(Tensor* tensor, float scalar, float* result_data) {
    
    for (int i = 0; i < tensor->size; i++) {
        result_data[i] = scalar * tensor->data[i];
    }
}

void scalar_div_tensor_cpu(float scalar, Tensor* tensor, float* result_data) {
    
    for (int i = 0; i < tensor->size; i++) {
        result_data[i] = scalar / tensor->data[i];
    }
}

void tensor_div_scalar_cpu(Tensor* tensor, float scalar, float* result_data) {
    
    for (int i = 0; i < tensor->size; i++) {
        result_data[i] = tensor->data[i] / scalar;
    }
}

void tensor_div_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data) {
    
    for (int i = 0; i < tensor1->size; i++) {
        result_data[i] = tensor1->data[i] / tensor2->data[i];
    }
}

void scalar_pow_tensor_cpu(float base, Tensor* tensor, float* result_data) {
    
    for (int i = 0; i < tensor->size; i++) {
        result_data[i] = powf(base, tensor->data[i]);
    }
}

void tensor_pow_scalar_cpu(Tensor* tensor, float exponent, float* result_data) {
    
    for (int i = 0; i < tensor->size; i++) {
        result_data[i] = powf(tensor->data[i], exponent);
    }
}

void log_tensor_cpu(Tensor* tensor, float* result_data) {
    for (int i = 0; i < tensor->size; i++) {
        result_data[i] = logf(tensor->data[i]);
    }
}

void sum_tensor_cpu(Tensor* tensor, float* result_data, int size, int* result_shape, int axis) {
    if (axis == -1) {
        // Sum over all elements
        float sum = 0.0;
        for (int i = 0; i < tensor->size; i++) {
            sum += tensor->data[i];
        }
        *result_data = sum;
    } else {
        if (axis < 0 || axis >= tensor->ndim) {
            printf("Invalid axis");
            return;
        }
        
        int axis_stride = tensor->strides[axis];

        for (int i = 0; i < tensor->shape[axis]; i++) {
            for (int j = 0; j < size; j++) {
                int index = 0;
                int remainder = j;
                for (int k = tensor->ndim - 2; k >= 0; k--) {
                    index += (remainder % result_shape[k]) * tensor->strides[k < axis ? k : k + 1];     
                    remainder /= result_shape[k];
                }
                result_data[j] += tensor->data[index + i * axis_stride];
            }
        }
    }
}

void max_tensor_cpu(Tensor* tensor, float* result_data, int size, int* result_shape, int axis) {
    if (axis == -1) {
        float max_value = -INFINITY;
        for (int i = 0; i < tensor->size; i++) {
            max_value = fmax(max_value, tensor->data[i]);
        }
        *result_data = max_value;
    } else {
        for (int i = 0; i < size; i++) {
            result_data[i] = -INFINITY;
        }
        if (axis < 0 || axis >= tensor->ndim) {
            printf("Invalid axis");
            return;
        }
        
        int axis_stride = tensor->strides[axis];

        for (int i = 0; i < tensor->shape[axis]; i++) {
            for (int j = 0; j < size; j++) {
                int index = 0;
                int remainder = j;
                for (int k = tensor->ndim - 2; k >= 0; k--) {
                    index += (remainder % result_shape[k]) * tensor->strides[k < axis ? k : k + 1];     
                    remainder /= result_shape[k];
                }
                result_data[j] = fmax(result_data[j], tensor->data[index + i * axis_stride]);
            }
        }
    }
}

void min_tensor_cpu(Tensor* tensor, float* result_data, int size, int* result_shape, int axis) {
    if (axis == -1) {
        float min_value = INFINITY;
        for (int i = 0; i < tensor->size; i++) {
            min_value = fmin(min_value, tensor->data[i]);
        }
        *result_data = min_value;
    } else {
        for (int i = 0; i < size; i++) {
            result_data[i] = INFINITY;
        }
        if (axis < 0 || axis >= tensor->ndim) {
            printf("Invalid axis");
            return;
        }
        
        int axis_stride = tensor->strides[axis];

        for (int i = 0; i < tensor->shape[axis]; i++) {
            for (int j = 0; j < size; j++) {
                int index = 0;
                int remainder = j;
                for (int k = tensor->ndim - 2; k >= 0; k--) {
                    index += (remainder % result_shape[k]) * tensor->strides[k < axis ? k : k + 1];     
                    remainder /= result_shape[k];
                }
                result_data[j] = fmin(result_data[j], tensor->data[index + i * axis_stride]);
            }
        }
    }
}

void equal_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data) {
    
    for (int i = 0; i < tensor1->size; i++) {
        result_data[i] = (tensor1->data[i] == tensor2->data[i]) ? 1.0f : 0.0f;
    }
}

void equal_broadcasted_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data, int* broadcasted_shape, int broadcasted_size) {
    int max_ndim = tensor1->ndim > tensor2->ndim ? tensor1->ndim : tensor2->ndim;

    // Calculate strides for broadcasting
    int* strides1 = (int*)malloc(max_ndim * sizeof(int));
    int* strides2 = (int*)malloc(max_ndim * sizeof(int));
    if (strides1 == NULL || strides2 == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    int stride1 = 1, stride2 = 1;
    for (int i = max_ndim - 1; i >= 0; i--) {
        int dim1 = i < tensor1->ndim ? tensor1->shape[tensor1->ndim - max_ndim + i] : 1;
        int dim2 = i < tensor2->ndim ? tensor2->shape[tensor2->ndim - max_ndim + i] : 1;
        strides1[i] = dim1 == broadcasted_shape[i] ? stride1 : 0;
        strides2[i] = dim2 == broadcasted_shape[i] ? stride2 : 0;
        stride1 *= (dim1 == broadcasted_shape[i]) ? dim1 : 1;
        stride2 *= (dim2 == broadcasted_shape[i]) ? dim2 : 1;
    }

    // Perform element-wise equal with broadcasting
    for (int i = 0; i < broadcasted_size; i++) {
        int index1 = 0, index2 = 0;
        int linear_index = i;
        for (int j = max_ndim - 1; j >= 0; j--) {
            int pos = linear_index % broadcasted_shape[j];
            linear_index /= broadcasted_shape[j];
            if (strides1[j] != 0) index1 += pos * strides1[j];
            if (strides2[j] != 0) index2 += pos * strides2[j];
        }
        result_data[i] = (tensor1->data[index1] == tensor2->data[index2]) ? 1.0f : 0.0f;
    }

    // Free strides
    free(strides1);
    free(strides2);
}


void ones_like_tensor_cpu(Tensor* tensor, float* result_data) {
    
    for (int i = 0; i < tensor->size; i++) {
        result_data[i] = 1.0;
    }
}

void zeros_like_tensor_cpu(Tensor* tensor, float* result_data) {
    
    for (int i = 0; i < tensor->size; i++) {
        result_data[i] = 0.0;
    }
}

void transpose_1D_tensor_cpu(Tensor* tensor, float* result_data) {

    for (int i = 0; i < tensor->shape[0]; i++) {
        result_data[i] = tensor->data[i];
    }
}

void transpose_2D_tensor_cpu(Tensor* tensor, float* result_data) {
    int rows = tensor->shape[0];
    int cols = tensor->shape[1];

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result_data[j * rows + i] = tensor->data[i * cols + j];
        }
    }
}

void transpose_3D_tensor_cpu(Tensor* tensor, float* result_data) {
    int batch = tensor->shape[0];
    int rows = tensor->shape[1];
    int cols = tensor->shape[2];

    for (int i = 0; i < batch; i++) {
        for (int j = 0; j < rows; j++) {
            for (int k = 0; k < cols; k++) {
                result_data[k * rows * batch + j * batch + i] = tensor->data[i * rows * cols + j * cols + k];
            }
        }
    }
}

void assign_tensor_cpu(Tensor* tensor, float* result_data) {

    for (int i = 0; i < tensor->size; i++) {
        result_data[i] = tensor->data[i];
    }
}

void make_contiguous_tensor_cpu(Tensor* tensor, float* result_data, int* new_strides) {
    
    for (int i = 0; i < tensor->size; i++) {
        int index = 0;
        int offset = i;
        for (int j = 0; j < tensor->ndim; j++) {
            index += (offset / new_strides[j]) * tensor->strides[j];
            offset %= new_strides[j];
        }
        result_data[i] = tensor->data[index];
    }

    // Free old data and update tensor properties
    free(tensor->data);
    free(tensor->strides);
    tensor->data = result_data;
    tensor->strides = new_strides;
}

void sin_tensor_cpu(Tensor* tensor, float* result_data) {
    for (int i = 0; i < tensor->size; i++) {
        result_data[i] = sinf(tensor->data[i]);
    }
}

void sigmoid_tensor_cpu(Tensor* tensor, float* result_data) {
    for (int i = 0; i < tensor->size; i++) {
        // avoid overflow
        if (tensor->data[i] >= 0) {

            float z = expf(-tensor->data[i]);
            result_data[i] = 1 / (1 + z);

        } else {

            float z = expf(tensor->data[i]);
            result_data[i] = z / (1 + z);
        }
    }
}

void cos_tensor_cpu(Tensor* tensor, float* result_data) {
    for (int i = 0; i < tensor->size; i++) {
        result_data[i] = cosf(tensor->data[i]);
    }
}

void matmul_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data) {
    
    for (int i = 0; i < tensor1->shape[0]; i++) {
        for (int j = 0; j < tensor2->shape[1]; j++) {
            float sum = 0.0;
            for (int k = 0; k < tensor1->shape[1]; k++) {
                sum += tensor1->data[i * tensor1->shape[1] + k] * tensor2->data[k * tensor2->shape[1] + j];
            }
            result_data[i * tensor2->shape[1] + j] = sum;
        }
    }
}


void broadcasted_batched_matmul_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data) {

    int result_data_stride = tensor1->shape[0] * tensor2->shape[2];

    for (int batch = 0; batch < tensor2->shape[0]; batch++) {
    
        for (int i = 0; i < tensor1->shape[0]; i++) {
            for (int j = 0; j < tensor2->shape[2]; j++) {
                float sum = 0.0;
                for (int k = 0; k < tensor1->shape[1]; k++) {
                    sum += tensor1->data[i * tensor1->shape[1] + k] * tensor2->data[batch*tensor2->strides[0] + (k * tensor2->shape[2] + j)];
                }
                result_data[(batch * result_data_stride) + (i * tensor2->shape[2] + j)] = sum;
            }
        }
    }
} 

void batched_matmul_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data) {
    int result_data_stride = tensor1->shape[1] * tensor2->shape[2];

    for (int batch = 0; batch < tensor2->shape[0]; batch++) {
    
        for (int i = 0; i < tensor1->shape[1]; i++) {
            for (int j = 0; j < tensor2->shape[2]; j++) {
                float sum = 0.0;
                for (int k = 0; k < tensor1->shape[2]; k++) {
                    sum += tensor1->data[(batch * tensor1->strides[0]) + i * tensor1->shape[2] + k] * tensor2->data[batch*tensor2->strides[0] + (k * tensor2->shape[2] + j)];
                }
                result_data[(batch * result_data_stride) + (i * tensor2->shape[2] + j)] = sum;
            }
        }
    }
} 

void conv2d_tensor_cpu(Tensor* input, Tensor* weight, Tensor* bias, float* result_data, int stride, int padding) {
    int batch_size = input->shape[0];
    int in_channels = input->shape[1];
    int in_height = input->shape[2];
    int in_width = input->shape[3];
    int out_channels = weight->shape[0];
    int kernel_height = weight->shape[2];
    int kernel_width = weight->shape[3];

    int out_height = (in_height + 2 * padding - kernel_height) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_width) / stride + 1;

    for (int b = 0; b < batch_size; b++) {
        for (int oc = 0; oc < out_channels; oc++) {
            for (int oh = 0; oh < out_height; oh++) {
                for (int ow = 0; ow < out_width; ow++) {
                    float sum = bias ? bias->data[oc] : 0.0f;
                    for (int ic = 0; ic < in_channels; ic++) {
                        for (int kh = 0; kh < kernel_height; kh++) {
                            for (int kw = 0; kw < kernel_width; kw++) {
                                int ih = oh * stride + kh - padding;
                                int iw = ow * stride + kw - padding;
                                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                    int input_idx = ((b * in_channels + ic) * in_height + ih) * in_width + iw;
                                    int weight_idx = ((oc * in_channels + ic) * kernel_height + kh) * kernel_width + kw;
                                    sum += input->data[input_idx] * weight->data[weight_idx];
                                }
                            }
                        }
                    }
                    int output_idx = ((b * out_channels + oc) * out_height + oh) * out_width + ow;
                    result_data[output_idx] = sum;
                }
            }
        }
    }
}

void qconv2d_tensor_cpu(Tensor* input, Tensor* weight, float* result_data, int stride, int padding) {

}