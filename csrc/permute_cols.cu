#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstdint>

static constexpr int default_threads = 256;
static constexpr int div_ceil(int a, int b) { return (a + b - 1) / b; }

// For a given "a" of size [M,K] performs a permutation of the K columns based
// on the given "perm" indices.
// Currently only supports 16bit types (since we permute half types)
__global__ void permute_cols_kernel(int4 const* __restrict__ a_int4_ptr,
                                    int const* __restrict__ perm_int_ptr,
                                    int4* __restrict__ out_int4_ptr, int size_m,
                                    int size_k, int block_rows) {
  int start_row = block_rows * blockIdx.x;
  int finish_row = start_row + block_rows;
  if (finish_row > size_m) {
    finish_row = size_m;
  }
  int cur_block_rows = std::max(finish_row - start_row, 0);

  int row_stride = size_k * sizeof(half) / 16;

  auto permute_row = [&](int row) {
    int iters = size_k / default_threads;
    int rest = size_k % default_threads;

    int offset = row * row_stride;

    half const* a_row_half = reinterpret_cast<half const*>(a_int4_ptr + offset);
    half* out_half = reinterpret_cast<half*>(out_int4_ptr + offset);

    int base_k = 0;

    for (int i = 0; i < iters; i++) {
      int cur_k = base_k + threadIdx.x;
      int src_pos = perm_int_ptr[cur_k];

      out_half[cur_k] = a_row_half[src_pos];

      base_k += default_threads;
    }

    if (rest) {
      if (threadIdx.x < rest) {
        int cur_k = base_k + threadIdx.x;
        int src_pos = perm_int_ptr[cur_k];

        out_half[cur_k] = a_row_half[src_pos];
      }
    }
  };

  for (int i = 0; i < cur_block_rows; i++) {
    int cur_row = start_row + i;
    if (cur_row < size_m) {
      permute_row(cur_row);
    }
  }
}

// More efficient version of A[..., perm]
//  taken from gptq_marlin.cu
torch::Tensor permute_cols(torch::Tensor const& A, torch::Tensor const& perm) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
  auto dev = A.get_device();
  auto stream = at::cuda::getCurrentCUDAStream(dev);

  TORCH_CHECK(A.scalar_type() == at::kHalf || A.scalar_type() == at::kBFloat16,
              "Currently only 16bit types are supported");
  TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
  TORCH_CHECK(A.size(-1) % 8 == 0,
              "A columns must be a multiple of 8 (128bits)");
  auto A_2d = A.view({-1, A.size(-1)});

  torch::Tensor D = torch::empty_like(A);
  int sms;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  int block_rows = div_ceil(A_2d.size(0), sms);
  permute_cols_kernel<<<sms, default_threads, 0, stream>>>(
      reinterpret_cast<int4 const*>(A_2d.const_data_ptr()),
      perm.const_data_ptr<int>(), reinterpret_cast<int4*>(D.mutable_data_ptr()),
      A_2d.size(0), A_2d.size(1), block_rows);
  return D;
}

__device__ __forceinline__ half multiply_scale(half a, half b) {
  return __float2half_rn(__half2float(a) * __half2float(b));
}

__device__ __forceinline__ __nv_bfloat16 multiply_scale(
    __nv_bfloat16 a, __nv_bfloat16 b) {
  return __float2bfloat16_rn(
      __bfloat162float(a) * __bfloat162float(b));
}

// Fuses the two Ours operations A[..., perm] and * scale. `perm` is int64
// because Ours checkpoints store it as a PyTorch LongTensor.
template <typename scalar_t>
__global__ void permute_and_scale_cols_kernel(
    scalar_t const* __restrict__ a_ptr,
    int64_t const* __restrict__ perm_ptr,
    scalar_t const* __restrict__ scale_ptr,
    scalar_t* __restrict__ out_ptr,
    int64_t numel,
    int64_t size_k) {
  for (int64_t index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < numel;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int64_t col = index % size_k;
    int64_t row_offset = index - col;
    out_ptr[index] = multiply_scale(
        a_ptr[row_offset + perm_ptr[col]], scale_ptr[col]);
  }
}

torch::Tensor permute_and_scale_cols(torch::Tensor const& A,
                                     torch::Tensor const& perm,
                                     torch::Tensor const& scale) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(A));

  TORCH_CHECK(A.scalar_type() == at::kHalf ||
                  A.scalar_type() == at::kBFloat16,
              "A must be float16 or bfloat16");
  TORCH_CHECK(perm.scalar_type() == at::kLong,
              "perm must be an int64 tensor");
  TORCH_CHECK(scale.scalar_type() == A.scalar_type(),
              "scale must have the same dtype as A");
  TORCH_CHECK(perm.is_contiguous() && scale.is_contiguous(),
              "perm and scale must be contiguous");
  TORCH_CHECK(A.dim() >= 1 && perm.dim() == 1 && scale.dim() == 1,
              "A must have a last dimension; perm and scale must be vectors");
  TORCH_CHECK(perm.numel() == A.size(-1) && scale.numel() == A.size(-1),
              "perm and scale must match A.size(-1)");
  TORCH_CHECK(A.get_device() == perm.get_device() &&
                  A.get_device() == scale.get_device(),
              "A, perm, and scale must be on the same GPU");

  torch::Tensor A_contiguous = A.contiguous();
  auto A_2d = A_contiguous.view({-1, A_contiguous.size(-1)});
  torch::Tensor D = torch::empty_like(A_contiguous);

  int64_t numel = A_2d.numel();
  int64_t size_k = A_2d.size(1);
  int blocks = static_cast<int>(
      (numel + default_threads - 1) / default_threads);
  auto stream = at::cuda::getCurrentCUDAStream(A.get_device());

  if (A.scalar_type() == at::kHalf) {
    permute_and_scale_cols_kernel<<<blocks, default_threads, 0, stream>>>(
        reinterpret_cast<half const*>(A_2d.const_data_ptr()),
        perm.const_data_ptr<int64_t>(),
        reinterpret_cast<half const*>(scale.const_data_ptr()),
        reinterpret_cast<half*>(D.mutable_data_ptr()),
        numel, size_k);
  } else {
    permute_and_scale_cols_kernel<<<blocks, default_threads, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16 const*>(A_2d.const_data_ptr()),
        perm.const_data_ptr<int64_t>(),
        reinterpret_cast<__nv_bfloat16 const*>(scale.const_data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(D.mutable_data_ptr()),
        numel, size_k);
  }
  return D;
}
