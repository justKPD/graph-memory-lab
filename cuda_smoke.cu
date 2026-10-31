#include <cuda_runtime.h>
#include <cstdio>
__global__ void add_one(float* x) { int i=threadIdx.x; x[i]+=1.0f; }
int main() {
    float h[32]={}, *d=nullptr;
    if(cudaMalloc(&d,sizeof(h))!=cudaSuccess) return 1;
    if(cudaMemcpy(d,h,sizeof(h),cudaMemcpyHostToDevice)!=cudaSuccess) return 2;
    add_one<<<1,32>>>(d);
    if(cudaGetLastError()!=cudaSuccess || cudaDeviceSynchronize()!=cudaSuccess) return 3;
    if(cudaMemcpy(h,d,sizeof(h),cudaMemcpyDeviceToHost)!=cudaSuccess) return 4;
    cudaFree(d);
    for(float v:h) if(v!=1.0f) return 5;
    puts("PASS: compiled CUDA kernel executed and 32 outputs verified");
    return 0;
}
