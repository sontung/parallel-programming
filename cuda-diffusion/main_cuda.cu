#include <stdio.h>
#include <cuda.h>

__global__ void calculate_g_image_gpu(float* in, float* out, int w, int h){
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int j = x % w;
    int i = x / w;
    if (1 <= i && i < h && 1 <= j && j < w) {
        float val = pow((in[(i+1)*w+j]-in[(i-1)*w+j])/2, 2) + pow((in[i*w+j+1]-in[i*w+j-1])/2, 2);
        float lambda = 3.5;
        out[i*w+j] = exp(-pow(val, 2)/2/pow(lambda, 2));
    }
}

__global__ void copy_gpu(float* src, float* dest, int w, int h) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int j = x % w;
    int i = x / w;
    dest[i*w+j] = src[i*w+j];
}

__device__ float arithmetic_mean_gpu(float n1, float n2) {
    return (n1+n2)/2.0;
}

__global__ void apply_stencil_gpu(float* in, float* out, float* g, int w, int h, float time_step) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int j = x % w;
    int i = x / w;
    if (1 <= i && i < h && 1 <= j && j < w) {
        float val = in[i*w+j]*(1-time_step*(arithmetic_mean_gpu(g[i*w+j], g[(i+1)*w+j]) +
	          		                        arithmetic_mean_gpu(g[i*w+j], g[i*w+j+1]) +
	          		                        arithmetic_mean_gpu(g[i*w+j], g[i*w+j-1])));
        val += in[(i+1)*w+j]*time_step*arithmetic_mean_gpu(g[i*w+j], g[(i+1)*w+j]);
        val += in[(i-1)*w+j]*time_step*arithmetic_mean_gpu(g[i*w+j], g[(i-1)*w+j]);
        val += in[i*w+j+1]*time_step*arithmetic_mean_gpu(g[i*w+j], g[i*w+j+1]);
        val += in[i*w+j-1]*time_step*arithmetic_mean_gpu(g[i*w+j], g[i*w+j-1]);
        val = (val < 0   ? 0   : val);
        val = (val > 255 ? 255 : val);
        out[i*w+j] = val;
    }
}


void gpu_func(float* in, float* out, float* g_img, int w, int h, int n_iters){

    int device = 0;
    int n = w*h;
   
    cudaSetDevice(device);
    float* in_dev;
    float* out_dev;
    float* g_dev;

    cudaMallocManaged(&in_dev, n * sizeof(float));
    cudaMallocManaged(&out_dev, n * sizeof(float));
    cudaMallocManaged(&g_dev, n * sizeof(float));
    for (int i = 0; i < n; i++) in_dev[i] = in[i];

    dim3 blockDim(16);

    dim3 gridDim((n % blockDim.x) ? n / blockDim.x : n / blockDim.x + 1);
    for (int t=0; t<n_iters; t++) {
        calculate_g_image_gpu<<<gridDim, blockDim>>>(in_dev, g_dev, w, h);
        apply_stencil_gpu<<<gridDim, blockDim>>>(in_dev, out_dev, g_dev, w, h, t);
        copy_gpu<<<gridDim, blockDim>>>(out_dev, in_dev, w, h);
    }
    cudaDeviceSynchronize(); 
    printf("Done executing CUDA kernel\n");
    for (int i = 0; i < n; i++) out[i] = out_dev[i];
    for (int i = 0; i < n; i++) g_img[i] = g_dev[i];
    
        
    cudaFree(in_dev);
    cudaFree(out_dev);
    cudaFree(g_dev);
}
