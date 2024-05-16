#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define THREADS_PER_BLOCK 128
#define TILE_SIZE 32

__host__ void cpu_to_cuda(Tensor* tensor) {
    
    float* data_tmp;
    cudaMalloc((void **)&data_tmp, tensor->size * sizeof(float));
    cudaMemcpy(data_tmp, tensor->data, tensor->size * sizeof(float), cudaMemcpyHostToDevice);

    tensor->data = data_tmp;

    const char* device_str = "cuda";
    tensor->device = (char*)malloc(strlen(device_str) + 1);
    strcpy(tensor->device, device_str); 

    printf("Successfully sent tensor to: %s\n", tensor->device);
}

__host__ void cuda_to_cpu(Tensor* tensor) {
    float* data_tmp = (float*)malloc(tensor->size * sizeof(float));

    cudaMemcpy(data_tmp, tensor->data, tensor->size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(tensor->data);

    tensor->data = data_tmp;

    const char* device_str = "cpu";
    tensor->device = (char*)malloc(strlen(device_str) + 1);
    strcpy(tensor->device, device_str); 

    printf("Successfully sent tensor to: %s\n", tensor->device);
}

__global__ void add_tensor_cuda_kernel(float* data1, float* data2, float* result_data, int size) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        result_data[i] = data1[i] + data2[i];
    }
}

__host__ void add_tensor_cuda(Tensor* tensor1, Tensor* tensor2, float* result_data) {
    
    int number_of_blocks = (tensor1->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    add_tensor_cuda_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(tensor1->data, tensor2->data, result_data, tensor1->size);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaDeviceSynchronize();
}

__global__ void add_broadcasted_tensor_cuda_kernel(float* data1, float* data2, float* result_data, int* shape1, int* shape2, int* broadcasted_shape, int ndim1, int ndim2, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    int idx1 = 0, idx2 = 0;
    int stride1 = 1, stride2 = 1;
    int linear_idx = i;

    for (int j = max(ndim1, ndim2) - 1; j >= 0; j--) {
        int dim1 = j < ndim1 ? shape1[ndim1 - 1 - j] : 1;
        int dim2 = j < ndim2 ? shape2[ndim2 - 1 - j] : 1;
        int broadcasted_dim = broadcasted_shape[j];

        int pos = linear_idx % broadcasted_dim;
        linear_idx /= broadcasted_dim;

        if (dim1 > 1) {
            idx1 += pos * stride1;
        }
        if (dim2 > 1) {
            idx2 += pos * stride2;
        }

        stride1 *= dim1;
        stride2 *= dim2;
    }

    result_data[i] = data1[idx1] + data2[idx2];
}

__host__ void add_broadcasted_tensor_cuda(Tensor* tensor1, Tensor* tensor2, float* result_data, int* broadcasted_shape) {
    int size = tensor1->size;

    // Copy the shapes to device memory
    int* d_shape1;
    int* d_shape2;
    int* d_broadcasted_shape;
    int ndim1 = tensor1->ndim;
    int ndim2 = tensor2->ndim;

    cudaMalloc((void**)&d_shape1, ndim1 * sizeof(int));
    cudaMalloc((void**)&d_shape2, ndim2 * sizeof(int));
    cudaMalloc((void**)&d_broadcasted_shape, max(ndim1, ndim2) * sizeof(int));

    cudaMemcpy(d_shape1, tensor1->shape, ndim1 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_shape2, tensor2->shape, ndim2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_broadcasted_shape, broadcasted_shape, max(ndim1, ndim2) * sizeof(int), cudaMemcpyHostToDevice);

    int number_of_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    add_broadcasted_tensor_cuda_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(tensor1->data, tensor2->data, result_data, d_shape1, d_shape2, d_broadcasted_shape, ndim1, ndim2, size);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaDeviceSynchronize();

    cudaFree(d_shape1);
    cudaFree(d_shape2);
    cudaFree(d_broadcasted_shape);
}

__global__ void sum_tensor_cuda_kernel(float* data, float* result_data, int size) {
    __shared__ float partial_sum[THREADS_PER_BLOCK * sizeof(float)];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    partial_sum[tid] = (i < size) ? data[i] : 0;

    __syncthreads();

    // Perform block-wise reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            partial_sum[tid] += partial_sum[tid + s];
        }
        __syncthreads();
    }

    // Write block sum to global memory
    if (tid == 0) {
        result_data[blockIdx.x] = partial_sum[0];
    }
}

__host__ void sum_tensor_cuda(Tensor* tensor, float* result_data) {
    cudaMemcpy(result_data, tensor->data, tensor->size * sizeof(float), cudaMemcpyHostToDevice);

    int num_blocks = (tensor->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // First-level reduction
    sum_tensor_cuda_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(tensor->data, result_data, tensor->size);

    // If necessary, perform multiple levels of reduction
    while (num_blocks > 1) {
        int num_blocks_next = (num_blocks + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        sum_tensor_cuda_kernel<<<num_blocks_next, THREADS_PER_BLOCK>>>(result_data, result_data, num_blocks);
        num_blocks = num_blocks_next;
    }

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaDeviceSynchronize();
}

__global__ void sub_tensor_cuda_kernel(float* data1, float* data2, float* result_data, int size) {
   
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        result_data[i] = data1[i] - data2[i];
    }
}

__host__ void sub_tensor_cuda(Tensor* tensor1, Tensor* tensor2, float* result_data) {
    
    int number_of_blocks = (tensor1->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    sub_tensor_cuda_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(tensor1->data, tensor2->data, result_data, tensor1->size);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaDeviceSynchronize();
}

__global__ void elementwise_mul_tensor_cuda_kernel(float* data1, float* data2, float* result_data, int size) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        result_data[i] = data1[i] * data2[i];
    }
}

__host__ void elementwise_mul_tensor_cuda(Tensor* tensor1, Tensor* tensor2, float* result_data) {
    
    int number_of_blocks = (tensor1->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    elementwise_mul_tensor_cuda_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(tensor1->data, tensor2->data, result_data, tensor1->size);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaDeviceSynchronize();
}

__global__ void scalar_mul_tensor_cuda_kernel(float* data, float scalar, float* result_data, int size) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        result_data[i] = scalar * data[i];
    }
}

__host__ void scalar_mul_tensor_cuda(Tensor* tensor, float scalar, float* result_data) {
    
    int number_of_blocks = (tensor->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    scalar_mul_tensor_cuda_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(tensor->data, scalar, result_data, tensor->size);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaDeviceSynchronize();
}

__global__ void scalar_div_tensor_cuda_kernel(float scalar, float* data, float* result_data, int size) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        result_data[i] = scalar / data[i];
    }
}

__host__ void scalar_div_tensor_cuda(float scalar, Tensor* tensor, float* result_data) {
    
    int number_of_blocks = (tensor->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    scalar_div_tensor_cuda_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(scalar, tensor->data, result_data, tensor->size);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaDeviceSynchronize();
}

__global__ void tensor_div_scalar_cuda_kernel(float* data, float scalar, float* result_data, int size) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        result_data[i] = data[i] / scalar;
    }
}

__host__ void tensor_div_scalar_cuda(Tensor* tensor, float scalar, float* result_data) {
    
    int number_of_blocks = (tensor->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    tensor_div_scalar_cuda_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(tensor->data, scalar, result_data, tensor->size);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaDeviceSynchronize();
}

__global__ void tensor_div_tensor_cuda_kernel(float* data1, float* data2, float* result_data, int size) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        result_data[i] = data1[i] / data2[i];
    }
}

__host__ void tensor_div_tensor_cuda(Tensor* tensor1, Tensor* tensor2, float* result_data) {
    
    int number_of_blocks = (tensor1->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    tensor_div_tensor_cuda_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(tensor1->data, tensor2->data, result_data, tensor1->size);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaDeviceSynchronize();
}

/*__global__ void matmul_tensor_cuda_kernel(float* data1, float* data2, float* result_data, int rows1, int cols1, int cols2) {    

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows1 && col < cols2) {
        float sum = 0.0;
        for (int k = 0; k < cols1; k++) {
            sum += data1[row * cols1 + k] * data2[k * cols2 + col];
        }
        result_data[row * cols2 + col] = sum;
    }

}*/

__global__ void matmul_tensor_cuda_kernel(float* data1, float* data2, float* result_data, int rows1, int cols1, int cols2) {    

    // Shared memory for tiles
    __shared__ float tile1[TILE_SIZE][TILE_SIZE];
    __shared__ float tile2[TILE_SIZE][TILE_SIZE];

    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Output position
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0;

    // Iterate over tiles
    for (int i = 0; i < (cols1 + TILE_SIZE - 1) / TILE_SIZE; ++i) {

        // Load tiles into shared memory
        if (row < rows1 && i * TILE_SIZE + tx < cols1)
            tile1[ty][tx] = data1[row * cols1 + i * TILE_SIZE + tx];
        else
            tile1[ty][tx] = 0.0;

        if (col < cols2 && i * TILE_SIZE + ty < cols1)
            tile2[ty][tx] = data2[(i * TILE_SIZE + ty) * cols2 + col];
        else
            tile2[ty][tx] = 0.0;

        // Synchronize threads
        __syncthreads();

        // Accumulate sum
        for (int k = 0; k < TILE_SIZE; ++k)
            sum += tile1[ty][k] * tile2[k][tx];

        // Synchronize threads
        __syncthreads();
    }

    // Write result to global memory
    if (row < rows1 && col < cols2)
        result_data[row * cols2 + col] = sum;
}


__host__ void matmul_tensor_cuda(Tensor* tensor1, Tensor* tensor2, float* result_data) {
    
    int rows1 = tensor1->shape[0];
    int cols1 = tensor1->shape[1];
    int cols2 = tensor2->shape[1];

    dim3 threadsPerBlock(16, 16);
    dim3 number_of_blocks((cols2 + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows1 + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matmul_tensor_cuda_kernel<<<number_of_blocks, threadsPerBlock>>>(tensor1->data, tensor2->data, result_data, rows1, cols1, cols2);


    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaDeviceSynchronize();
}

__global__ void tensor_pow_scalar_cuda_kernel(float* data, float exponent, float* result_data, int size) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        result_data[i] = powf(data[i], exponent);
    }
}

__host__ void tensor_pow_scalar_cuda(Tensor* tensor, float exponent, float* result_data) {
    
    int number_of_blocks = (tensor->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    tensor_pow_scalar_cuda_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(tensor->data, exponent, result_data, tensor->size);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaDeviceSynchronize();
}

__global__ void scalar_pow_tensor_cuda_kernel(float base, float* data, float* result_data, int size) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        result_data[i] = powf(base, data[i]);
    }
}

__host__ void scalar_pow_tensor_cuda(float base, Tensor* tensor, float* result_data) {
    
    int number_of_blocks = (tensor->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    scalar_pow_tensor_cuda_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(base, tensor->data, result_data, tensor->size);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaDeviceSynchronize();
}

__global__ void log_tensor_cuda_kernel(float* data, float* result_data, int size) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        result_data[i] = logf(data[i]);
    }
}

__host__ void log_tensor_cuda(Tensor* tensor, float* result_data) {
    
    int number_of_blocks = (tensor->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    log_tensor_cuda_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(tensor->data, result_data, tensor->size);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaDeviceSynchronize();
}


__global__ void ones_like_tensor_cuda_kernel(float* data, float* result_data, int size) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        result_data[i] = 1.0;
    }
}

__host__ void ones_like_tensor_cuda(Tensor* tensor, float* result_data) {
    
    int number_of_blocks = (tensor->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    ones_like_tensor_cuda_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(tensor->data, result_data, tensor->size);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaDeviceSynchronize();
}

__global__ void zeros_like_tensor_cuda_kernel(float* data, float* result_data, int size) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        result_data[i] = 0.0;
    }
}

__host__ void zeros_like_tensor_cuda(Tensor* tensor, float* result_data) {
    
    int number_of_blocks = (tensor->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    zeros_like_tensor_cuda_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(tensor->data, result_data, tensor->size);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaDeviceSynchronize();
}

__global__ void transpose_tensor_cuda_kernel(float* data, float* result_data, int rows, int cols) {
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (tid_x < cols && tid_y < rows) {
        result_data[tid_x * rows + tid_y] = data[tid_y * cols + tid_x];
    }
}

__host__ void transpose_tensor_cuda(Tensor* tensor, float* result_data) {
    
    int rows = tensor->shape[0];
    int cols = tensor->shape[1];

    dim3 threadsPerBlock(16, 16);
    dim3 number_of_blocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
    transpose_tensor_cuda_kernel<<<number_of_blocks, threadsPerBlock>>>(tensor->data, result_data, rows, cols);


    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaDeviceSynchronize();
}

__global__ void assign_tensor_cuda_kernel(float* data, float* result_data, int size) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        result_data[i] = data[i];
    }
}

__host__ void assign_tensor_cuda(Tensor* tensor, float* result_data) {
    
    int number_of_blocks = (tensor->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    assign_tensor_cuda_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(tensor->data, result_data, tensor->size);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaDeviceSynchronize();
}

__global__ void sin_tensor_cuda_kernel(float* data, float* result_data, int size) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        result_data[i] = sinf(data[i]);
    }
}

__host__ void sin_tensor_cuda(Tensor* tensor, float* result_data) {
    
    int number_of_blocks = (tensor->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    sin_tensor_cuda_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(tensor->data, result_data, tensor->size);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaDeviceSynchronize();
}

__global__ void cos_tensor_cuda_kernel(float* data, float* result_data, int size) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        result_data[i] = cosf(data[i]);
    }
}

__host__ void cos_tensor_cuda(Tensor* tensor, float* result_data) {
    
    int number_of_blocks = (tensor->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cos_tensor_cuda_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(tensor->data, result_data, tensor->size);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaDeviceSynchronize();
}




