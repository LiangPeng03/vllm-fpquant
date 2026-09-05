#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <tuple>

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

// The NADA/Ours deployment path has an identity Hadamard transform. This
// operator implements its online LSS activation quantization in the same
// row-major FP4/E4M3 layout consumed by the existing NVFP4 GEMM.
//
// One warp handles one consecutive group of 16 BF16 activations.  The first
// kernel computes the unrounded LSS scale; PyTorch performs the E4M3
// round-trip so its conversion semantics stay identical to exported weights.
// The second kernel uses that rounded scale to pack the two E2M1 values per
// byte. This is intentionally separate from Qutlass because no matrix
// transform is required for Ours.
__device__ __forceinline__ float fp4_round_magnitude(float x) {
  if (x > 5.0f) {
    return 6.0f;
  }
  if (x >= 3.5f) {
    return 4.0f;
  }
  if (x >= 1.75f) {
    return static_cast<float>(__float2int_rn(x));
  }
  return 0.5f * static_cast<float>(__float2int_rn(2.0f * x));
}

// The calibration-side Quantizer keeps NVFP4 LSS intermediates in BF16.
// Round explicitly at each tensor-operation boundary so the online path uses
// the same grid assignments and E4M3 scale inputs as pseudoquantization.
__device__ __forceinline__ float round_to_bf16(float value) {
  return __bfloat162float(__float2bfloat16_rn(value));
}

__device__ __forceinline__ uint8_t encode_fp4(float value) {
  float magnitude = fp4_round_magnitude(fabsf(value));
  int code;
  if (magnitude == 0.0f) {
    // Qutlass's activation quantizer stores both exact and rounded zeros as
    // positive zero. Weight-export packing has a different signed-zero tie
    // convention, so activation inputs must follow Qutlass here.
    code = 0;
  } else if (magnitude == 0.5f) {
    code = 1;
  } else if (magnitude == 1.0f) {
    code = 2;
  } else if (magnitude == 1.5f) {
    code = 3;
  } else if (magnitude == 2.0f) {
    code = 4;
  } else if (magnitude == 3.0f) {
    code = 5;
  } else if (magnitude == 4.0f) {
    code = 6;
  } else {
    code = 7;
  }
  return static_cast<uint8_t>(value < 0.0f ? code | 8 : code);
}

__global__ void nvfp4_lss_scales_kernel(
    __nv_bfloat16 const* __restrict__ input,
    __nv_bfloat16* __restrict__ scales,
    int64_t rows, int64_t groups_per_row, int64_t padded_groups_per_row) {
  constexpr int warps_per_block = default_threads / 32;
  int warp = threadIdx.x / 32;
  int lane = threadIdx.x % 32;
  int64_t group_index =
      static_cast<int64_t>(blockIdx.x) * warps_per_block + warp;
  int64_t group_count = rows * groups_per_row;
  if (group_index >= group_count) {
    return;
  }

  int64_t row = group_index / groups_per_row;
  int64_t group = group_index % groups_per_row;
  float abs_x = 0.0f;
  if (lane < 16) {
    abs_x = fabsf(__bfloat162float(input[row * groups_per_row * 16 +
                                         group * 16 + lane]));
  }

  float max_abs = abs_x;
  for (int offset = 16; offset > 0; offset /= 2) {
    max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffff, max_abs, offset));
  }
  float initial_scale =
      round_to_bf16(__shfl_sync(0xffffffff, max_abs, 0) / 6.0f);
  float safe_initial_scale = initial_scale == 0.0f ? 1.0f : initial_scale;
  float grid = lane < 16
                   ? fp4_round_magnitude(
                         round_to_bf16(abs_x / safe_initial_scale))
                   : 0.0f;
  float numerator = lane < 16 ? round_to_bf16(abs_x * grid) : 0.0f;
  float denominator = lane < 16 ? round_to_bf16(grid * grid) : 0.0f;
  for (int offset = 16; offset > 0; offset /= 2) {
    numerator += __shfl_down_sync(0xffffffff, numerator, offset);
    denominator += __shfl_down_sync(0xffffffff, denominator, offset);
  }
  if (lane == 0) {
    // The reference keeps its original (zero) scale for an all-zero group;
    // the post-E4M3 stage below then changes it to one.
    numerator = round_to_bf16(numerator);
    denominator = round_to_bf16(denominator);
    float lss_scale = denominator == 0.0f
                          ? initial_scale
                          : round_to_bf16(numerator / denominator);
    scales[row * padded_groups_per_row + group] = __float2bfloat16_rn(lss_scale);
  }
}

__global__ void nvfp4_lss_pack_kernel(
    __nv_bfloat16 const* __restrict__ input,
    float const* __restrict__ rounded_scales,
    float const* __restrict__ global_scale, uint8_t* __restrict__ output,
    int64_t rows, int64_t groups_per_row, int64_t padded_groups_per_row) {
  constexpr int warps_per_block = default_threads / 32;
  int warp = threadIdx.x / 32;
  int lane = threadIdx.x % 32;
  int64_t group_index =
      static_cast<int64_t>(blockIdx.x) * warps_per_block + warp;
  int64_t group_count = rows * groups_per_row;
  if (group_index >= group_count || lane >= 8) {
    return;
  }

  int64_t row = group_index / groups_per_row;
  int64_t group = group_index % groups_per_row;
  float local_scale =
      rounded_scales[row * padded_groups_per_row + group] / global_scale[0];
  local_scale = round_to_bf16(local_scale);
  if (local_scale == 0.0f) {
    local_scale = 1.0f;
  }

  int64_t input_offset = row * groups_per_row * 16 + group * 16 + lane * 2;
  // Quantizer.quantize performs the activation/scale division in BF16 before
  // cast_to_fp4. Preserve that rounding boundary before selecting an E2M1 bin.
  uint8_t lo = encode_fp4(round_to_bf16(
      __bfloat162float(input[input_offset]) / local_scale));
  uint8_t hi = encode_fp4(round_to_bf16(
      __bfloat162float(input[input_offset + 1]) / local_scale));
  output[row * groups_per_row * 8 + group * 8 + lane] = lo | (hi << 4);
}

std::tuple<torch::Tensor, torch::Tensor> nvfp4_lss_quant_identity(
    torch::Tensor const& A, torch::Tensor const& global_scale) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
  TORCH_CHECK(A.scalar_type() == at::kBFloat16, "A must be bfloat16");
  TORCH_CHECK(A.is_contiguous() && A.dim() == 2 && A.size(1) % 16 == 0,
              "A must be a contiguous [rows, cols] tensor with cols divisible by 16");
  TORCH_CHECK(global_scale.scalar_type() == at::kFloat &&
                  global_scale.is_contiguous() && global_scale.numel() == 1,
              "global_scale must be a contiguous float32 scalar");
  TORCH_CHECK(A.get_device() == global_scale.get_device(),
              "A and global_scale must be on the same GPU");

  int64_t rows = A.size(0);
  int64_t groups_per_row = A.size(1) / 16;
  int64_t padded_rows = ((rows + 127) / 128) * 128;
  int64_t padded_groups_per_row = ((groups_per_row + 3) / 4) * 4;
  auto options = A.options();
  torch::Tensor raw_scales = torch::zeros(
      {padded_rows, padded_groups_per_row}, options.dtype(at::kBFloat16));
  torch::Tensor output = torch::empty(
      {rows, A.size(1) / 2}, options.dtype(at::kByte));

  int64_t group_count = rows * groups_per_row;
  int blocks = static_cast<int>((group_count + 7) / 8);
  auto stream = at::cuda::getCurrentCUDAStream(A.get_device());
  nvfp4_lss_scales_kernel<<<blocks, default_threads, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16 const*>(A.const_data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(raw_scales.mutable_data_ptr()), rows, groups_per_row,
      padded_groups_per_row);

  // Preserve the E4M3 round-trip used by Quantizer.get_quantization_params.
  torch::Tensor fp8_scales =
      (raw_scales * global_scale).clamp(-448.0, 448.0).to(at::kFloat8_e4m3fn);
  torch::Tensor rounded_scales = fp8_scales.to(at::kFloat);

  nvfp4_lss_pack_kernel<<<blocks, default_threads, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16 const*>(A.const_data_ptr()),
      rounded_scales.const_data_ptr<float>(), global_scale.const_data_ptr<float>(),
      output.mutable_data_ptr<uint8_t>(), rows, groups_per_row,
      padded_groups_per_row);
  return std::make_tuple(output, fp8_scales);
}
