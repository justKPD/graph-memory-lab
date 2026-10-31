#include <torch/types.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

template<typename T> __global__ void segmented_gather_sum(
 const T* __restrict__ x, const int64_t* __restrict__ ptr,
 const int64_t* __restrict__ col, T* __restrict__ y, int64_t n, int64_t f) {
  const int64_t i=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=n*f) return;
  const int64_t row=i/f, feature=i%f;
  T value=0;
  for(int64_t edge=ptr[row]; edge<ptr[row+1]; ++edge)
    value += x[col[edge]*f+feature];
  y[i]=value;
}

torch::Tensor csr_sum_cuda(torch::Tensor x, torch::Tensor ptr, torch::Tensor col) {
  const c10::cuda::CUDAGuard guard(x.device());
  auto y=torch::empty_like(x);
  const auto count=x.numel();
  if(count==0) return y;
  const auto stream=c10::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "gml_csr_sum", [&] {
    segmented_gather_sum<scalar_t><<<(count+255)/256,256,0,stream>>>(
      x.data_ptr<scalar_t>(),ptr.data_ptr<int64_t>(),col.data_ptr<int64_t>(),
      y.data_ptr<scalar_t>(),x.size(0),x.size(1));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
