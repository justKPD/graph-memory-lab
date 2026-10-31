"""Compile a tiny PyTorch C++/CUDA extension and verify device execution."""
from pathlib import Path
import json
import torch
from torch.utils.cpp_extension import load_inline

cpp = r'''
#include <torch/extension.h>
void launch_add_one(float* data, int n);
torch::Tensor add_one(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type()==torch::kFloat32 && x.is_contiguous());
  TORCH_CHECK(x.numel()<=1024);
  auto y=x.clone();
  launch_add_one(y.data_ptr<float>(),static_cast<int>(y.numel()));
  return y;
}
'''
cuda = r'''
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
__global__ void add(float* x,int n) { int i=threadIdx.x; if(i<n) x[i]+=1.f; }
void launch_add_one(float* x,int n) {
  add<<<1,1024,0,c10::cuda::getCurrentCUDAStream()>>>(x,n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
'''
extension = load_inline(name='graph_memory_lab_setup_smoke', cpp_sources=cpp,
                        cuda_sources=cuda, functions=['add_one'],
                        extra_cuda_cflags=['-O2'], verbose=True)
x=torch.arange(32,device='cuda',dtype=torch.float32)
torch.testing.assert_close(extension.add_one(x),x+1)
torch.cuda.synchronize()
result={'extension_compile_and_execution':'PASS','elements_checked':32,
        'is_graph_kernel':False,'is_benchmark':False}
Path('extension_verified.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
