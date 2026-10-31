#include <torch/extension.h>
torch::Tensor csr_sum_cuda(torch::Tensor x, torch::Tensor ptr, torch::Tensor col);
torch::Tensor csr_sum(torch::Tensor x, torch::Tensor ptr, torch::Tensor col) {
  TORCH_CHECK(x.is_cuda() && ptr.is_cuda() && col.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(x.device()==ptr.device() && x.device()==col.device(), "Device mismatch");
  TORCH_CHECK(x.dim()==2 && ptr.dim()==1 && col.dim()==1, "Invalid rank");
  TORCH_CHECK(ptr.numel()==x.size(0)+1, "Invalid CSR pointer size");
  TORCH_CHECK(ptr.scalar_type()==torch::kInt64 && col.scalar_type()==torch::kInt64, "int64 indices required");
  TORCH_CHECK(x.is_contiguous() && ptr.is_contiguous() && col.is_contiguous(), "Contiguous inputs required");
  TORCH_CHECK(x.scalar_type()==torch::kFloat32 || x.scalar_type()==torch::kFloat64, "Unsupported precision");
  return csr_sum_cuda(x,ptr,col);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("csr_sum", &csr_sum); }
