# SPDX-License-Identifier: Apache-2.0
"""ScaleSweep NVFP4 activation quantization matching MR-GPTQ checkpoints.

The global scale is the inverse physical tensor scale. Search uses the floor
E4M3 code, offsets [-3,7] for MSE or [-8,7] for WMSE, and original-unit errors.
Outputs use FPQuant's row-major padded scales and standard E2M1 packed bytes.
"""

from typing import Optional

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _fp4(value):
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .b8 code;
          .reg .b32 halves;
          .reg .b16 lo, hi;
          cvt.rn.satfinite.e2m1x2.f32 code, $1, $1;
          cvt.rn.f16x2.e2m1x2 halves, code;
          mov.b32 {lo, hi}, halves;
          cvt.f32.f16 $0, lo;
        }
        """,
        constraints="=f,f", args=[value], dtype=tl.float32,
        is_pure=True, pack=1,
    )


@triton.jit
def _pack_pair(low, high):
    packed = tl.inline_asm_elementwise(
        asm="""
        {
          .reg .b8 code;
          cvt.rn.satfinite.e2m1x2.f32 code, $2, $1;
          cvt.u32.u8 $0, code;
        }
        """,
        constraints="=r,f,f", args=[low, high], dtype=tl.uint32,
        is_pure=True, pack=1,
    )
    # The MR-GPTQ packer canonicalizes both signed zeros to code 0.
    packed = tl.where((packed & 15) == 8, packed & 240, packed)
    return tl.where((packed >> 4) == 8, packed & 15, packed)


@triton.jit
def _quantize(
    X, Global, Importance, Packed, Scales, group_count, groups_per_row,
    padded_groups, WEIGHTED: tl.constexpr, BLOCKS: tl.constexpr,
):
    group = tl.program_id(0) * BLOCKS + tl.arange(0, BLOCKS)
    channel = tl.arange(0, 16)
    candidate = tl.arange(0, 16)
    values = tl.load(
        X + group[:, None] * 16 + channel[None, :],
        mask=group[:, None] < group_count, other=0,
    ).to(tl.float32)
    inverse = tl.load(Global).to(tl.float32)
    base = tl.minimum(tl.max(tl.abs(values), axis=1) * inverse / 6.0, 448.0)
    bits = base.to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 255) - 120
    mantissa = (bits & 0x7fffff) >> 20
    base_raw = tl.where(exponent > 0, (exponent << 3) + mantissa,
                        tl.floor(base * 512.0).to(tl.int32))
    base_raw = tl.minimum(tl.maximum(base_raw, 0), 126)
    lower = -8 if WEIGHTED else -3
    offsets = candidate + lower
    raw = tl.minimum(tl.maximum(base_raw[:, None] + offsets[None, :], 1), 126)
    scale_exp, scale_man = raw >> 3, raw & 7
    fp32 = ((scale_exp + 120) << 23) | (scale_man << 20)
    decoded = tl.where(scale_exp > 0, fp32.to(tl.float32, bitcast=True),
                       scale_man.to(tl.float32) / 512.0)
    physical = decoded / inverse
    reconstructed = _fp4(values[:, :, None] / physical[:, None, :]) * physical[:, None, :]
    error = reconstructed - values[:, :, None]
    error = error * error
    if WEIGHTED:
        importance = tl.load(
            Importance + (group[:, None] % groups_per_row) * 16 + channel[None, :],
            mask=group[:, None] < group_count, other=0,
        ).to(tl.float32)
        error = error * importance[:, :, None]
    loss = tl.sum(error, axis=1) / 16.0
    loss = tl.where(offsets[None, :] <= 7, loss, float("inf"))
    minimum = tl.min(loss, axis=1)
    winner = tl.min(tl.where(loss == minimum[:, None], candidate[None, :], 16), axis=1)
    selected_raw = tl.sum(tl.where(candidate[None, :] == winner[:, None], raw, 0), axis=1)
    selected_scale = tl.sum(
        tl.where(candidate[None, :] == winner[:, None], physical, 0.0), axis=1
    )
    pairs = tl.reshape(values / selected_scale[:, None], (BLOCKS, 8, 2))
    component = tl.arange(0, 2)
    low = tl.sum(tl.where(component[None, None, :] == 0, pairs, 0), axis=2)
    high = tl.sum(tl.where(component[None, None, :] == 1, pairs, 0), axis=2)
    tl.store(
        Packed + group[:, None] * 8 + tl.arange(0, 8)[None, :],
        _pack_pair(low, high).to(tl.uint8),
        mask=group[:, None] < group_count,
    )
    row = group // groups_per_row
    column = group % groups_per_row
    tl.store(Scales + row * padded_groups + column, selected_raw.to(tl.uint8),
             mask=group < group_count)


def scalesweep_quantize(
    x: torch.Tensor, global_scale: torch.Tensor,
    importance: Optional[torch.Tensor], weighted: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (x.ndim != 2 or not x.is_cuda or not x.is_contiguous()
            or x.dtype != torch.bfloat16 or x.shape[1] % 16):
        raise ValueError("ScaleSweep requires contiguous CUDA BF16 [M,K], K divisible by 16.")
    if (global_scale.dtype != torch.float32 or global_scale.numel() != 1
            or not global_scale.is_contiguous() or global_scale.device != x.device):
        raise ValueError("ScaleSweep requires a contiguous CUDA FP32 global scale.")
    if weighted and (
        importance is None or importance.dtype != torch.float32
        or importance.device != x.device or not importance.is_contiguous()
        or importance.ndim != 1 or importance.numel() != x.shape[1]
    ):
        raise ValueError("WMSE requires the checkpoint's CUDA FP32 channel importance.")
    rows, width = x.shape
    groups = width // 16
    packed = torch.empty((rows, width // 2), device=x.device, dtype=torch.uint8)
    scales = torch.zeros(
        ((rows + 127) // 128 * 128, (groups + 3) // 4 * 4),
        device=x.device, dtype=torch.float8_e4m3fn,
    )
    if rows == 0:
        return packed, scales
    _quantize[(triton.cdiv(rows * groups, 16),)](
        x, global_scale, importance if weighted else x,
        packed, scales.view(torch.uint8), rows * groups, groups, scales.shape[1],
        WEIGHTED=weighted, BLOCKS=16, num_warps=4, enable_fp_fusion=False,
    )
    return packed, scales


def scalesweep_quantize_fake(x, global_scale, importance, weighted):
    rows, width = x.shape
    return (
        torch.empty((rows, width // 2), device=x.device, dtype=torch.uint8),
        torch.empty(((rows + 127) // 128 * 128, (width // 16 + 3) // 4 * 4),
                    device=x.device, dtype=torch.float8_e4m3fn),
    )


@torch.no_grad()
def reference_quantize(x, global_scale, importance=None):
    """Independent PyTorch candidate sweep, returning packed bytes and scales."""
    rows, width = x.shape
    values = x.float().reshape(-1, 16)
    table = torch.arange(127, device=x.device, dtype=torch.uint8)
    table = table.view(torch.float8_e4m3fn).float()
    base = (values.abs().amax(-1) * global_scale / 6).clamp(0, 448)
    base = (torch.searchsorted(table, base.contiguous(), right=True) - 1).clamp(0, 126)
    offsets = torch.arange(-3 if importance is None else -8, 8, device=x.device)
    raw = (base[:, None] + offsets).clamp(1, 126)
    scale = table[raw] / global_scale
    normalized = values[:, :, None] / scale[:, None, :]
    magnitude = normalized.abs()
    rounded = torch.where(
        magnitude > 5, 6,
        torch.where(magnitude >= 3.5, 4,
                    torch.where(magnitude >= 1.75, magnitude.round(),
                                (magnitude * 2).round() * 0.5)),
    ) * normalized.sign()
    error = (rounded * scale[:, None, :] - values[:, :, None]).square()
    if importance is not None:
        weights = importance.reshape(-1, 16).repeat(rows, 1)
        error = error * weights[:, :, None]
    winner = error.mean(1).argmin(-1, keepdim=True)
    selected_raw = raw.gather(1, winner).squeeze(1)
    selected_scale = scale.gather(1, winner).squeeze(1)
    selected = rounded.gather(2, winner[:, None, :].expand(-1, 16, -1)).squeeze(-1)
    grid = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=x.device)
    codes = torch.bucketize(selected.abs().contiguous(), grid).to(torch.uint8)
    codes = codes | torch.where(selected < 0, 8, 0).to(torch.uint8)
    codes = codes.reshape(rows, width)
    packed = codes[:, ::2] | (codes[:, 1::2] << 4)
    scales = torch.zeros(((rows + 127) // 128 * 128, (width // 16 + 3) // 4 * 4),
                         device=x.device, dtype=torch.float8_e4m3fn)
    scales.view(torch.uint8)[:rows, :width // 16] = selected_raw.reshape(rows, width // 16).to(torch.uint8)
    reconstructed = (selected * selected_scale[:, None]).reshape(rows, width).to(x.dtype)
    return packed, scales, reconstructed


@torch.no_grad()
def validate_runtime():
    """Check packing, real WMSE, CUTLASS output, loading, and CUDA Graph replay."""
    from torch.nn import functional as F
    from vllm.model_executor.layers.quantization.fp_quant import (
        _load_scalesweep_importance,
    )

    torch.manual_seed(0)
    width = 2048
    global_scale = torch.tensor([2.75], device="cuda", dtype=torch.float32)
    for weighted in (False, True):
        importance = (
            torch.exp(torch.linspace(-3, 3, width, device="cuda")) if weighted else None
        )
        for rows in (1, 16, 128):
            x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
            x[:, :16] = 0
            x[:, 16:32] *= 10000
            x[:, 32:48] *= 0.0001
            actual, scales = torch.ops.vllm.nvfp4_scalesweep_quant(
                x, global_scale, importance, weighted
            )
            expected, expected_scales, _ = reference_quantize(x, global_scale, importance)
            assert torch.equal(actual, expected), f"FP4 packing mismatch: {weighted}, {rows}"
            assert torch.equal(scales.view(torch.uint8), expected_scales.view(torch.uint8)), (
                f"FP8 scale mismatch: {weighted}, {rows}"
            )
        x = torch.randn(16, width, device="cuda", dtype=torch.bfloat16)
        qx, xs = torch.ops.vllm.nvfp4_scalesweep_quant(x, global_scale, importance, weighted)
        _, _, dx = reference_quantize(x, global_scale, importance)
        weight = torch.randn(128, width, device="cuda", dtype=torch.bfloat16)
        weight_global = torch.tensor([3.5], device="cuda", dtype=torch.float32)
        qw, ws, dw = reference_quantize(weight, weight_global)
        actual = torch.ops.vllm.matmul_nvf4_bf16(
            qx, qw, xs, ws, 1 / (global_scale * weight_global)
        )
        expected = F.linear(dx.float(), dw.float())
        relative_l2 = (actual.float() - expected).norm() / expected.norm().clamp_min(1e-20)
        assert relative_l2.item() < 0.02, f"CUTLASS relative L2 error: {relative_l2.item()}"
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured, captured_scales = torch.ops.vllm.nvfp4_scalesweep_quant(
                x, global_scale, importance, weighted
            )
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured, qx)
        assert torch.equal(captured_scales.view(torch.uint8), xs.view(torch.uint8))
        mode = "WMSE" if weighted else "MSE"
        print(f"ScaleSweep-{mode}: packing/scales/CUTLASS/CUDA Graph OK, "
              f"relative_L2={relative_l2.item():.6g}", flush=True)
    importance = torch.ones(width, device="cuda", dtype=torch.float32)
    importance[::2] = 100.0
    x = torch.randn(128, width, device="cuda", dtype=torch.bfloat16)
    uniform_codes, _ = scalesweep_quantize(
        x, global_scale, torch.ones_like(importance), True
    )
    weighted_codes, _ = scalesweep_quantize(x, global_scale, importance, True)
    assert not torch.equal(uniform_codes, weighted_codes), "WMSE importance has no effect"
    parameter = torch.nn.Parameter(torch.full_like(importance, float("nan")), requires_grad=False)
    _load_scalesweep_importance(parameter, importance, "q")
    _load_scalesweep_importance(parameter, importance.clone(), "k")
    try:
        _load_scalesweep_importance(parameter, importance + 1, "v")
    except ValueError:
        pass
    else:
        raise AssertionError("Mismatched fused importance vectors must be rejected")
    print("ScaleSweep WMSE importance loading/effect: OK", flush=True)
