# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Supports FP-Quant compression, see https://arxiv.org/abs/2509.23202

import os
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from vllm._custom_ops import (
    cutlass_scaled_fp4_mm,
    fusedQuantizeMx,
    fusedQuantizeNv,
    matmul_mxf4_bf16_tn,
    nvfp4_lss_quant_identity,
    nvfp4_lss_quant_permute_scale,
    nvfp4_lss_quant_permute_scale_cutlass,
    permute_and_scale_cols,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.qutlass_utils import to_blocked
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.utils import direct_register_custom_op
from vllm.model_executor.layers.quantization.utils.scalesweep import (
    scalesweep_quantize, scalesweep_quantize_fake,
)


def _load_scalesweep_importance(param, loaded_weight, shard_id=None):
    """Whole input-channel vector shared by fused QKV and gate/up shards."""
    value = loaded_weight.to(device=param.device, dtype=torch.float32).reshape(-1)
    if value.shape != param.shape:
        raise ValueError("ScaleSweep importance width does not match this layer.")
    if getattr(param, "_scalesweep_loaded", False) and not torch.equal(param.data, value):
        raise ValueError("Fused ScaleSweep shards have different activation importance.")
    param.data.copy_(value)
    param._scalesweep_loaded = True


class FPQuantConfig(QuantizationConfig):
    """Config class for FPQuant."""

    def __init__(
        self,
        hadamard_group_size: int = 32,
        forward_dtype: str = "mxfp4",
        forward_method: str = "abs_max",
        lss_identity_transform: bool = False,
        pseudoquantization: bool = False,
        modules_to_not_convert: Optional[list[str]] = None,
        scalesweep_identity_transform: bool = False,
    ) -> None:
        super().__init__()
        self.hadamard_group_size = hadamard_group_size
        self.forward_dtype = forward_dtype
        self.forward_method = forward_method
        self.lss_identity_transform = lss_identity_transform
        self.pseudoquantization = pseudoquantization
        self.modules_to_not_convert = modules_to_not_convert
        self.scalesweep_identity_transform = scalesweep_identity_transform

        if pseudoquantization:
            raise ValueError("Pseudoquantization is not supported for vLLM")
        if forward_method == "lss" and (
            forward_dtype != "nvfp4"
            or hadamard_group_size != 16
            or not lss_identity_transform
        ):
            raise ValueError(
                "LSS runtime quantization currently supports only NVFP4, "
                "group size 16, and an identity transform."
            )
        if forward_method.startswith("scalesweep_") and (
            forward_method not in {"scalesweep_mse", "scalesweep_wmse"}
            or forward_dtype != "nvfp4" or hadamard_group_size != 16
            or not scalesweep_identity_transform
        ):
            raise ValueError("ScaleSweep supports NVFP4 group-16 identity-transform checkpoints.")

    def __repr__(self) -> str:
        return (
            f"FPQuantConfig(hadamard_group_size={self.hadamard_group_size}, "
            f"forward_dtype={self.forward_dtype}, "
            f"forward_method={self.forward_method}, "
            f"lss_identity_transform={self.lss_identity_transform}, "
            f"scalesweep_identity_transform={self.scalesweep_identity_transform}, "
            f"pseudoquantization={self.pseudoquantization}, "
            f"modules_to_not_convert={self.modules_to_not_convert})"
        )

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "fp_quant"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 100

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []  # no extra configs.

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "FPQuantConfig":
        hadamard_group_size = cls.get_from_keys(config, ["hadamard_group_size"])
        forward_dtype = cls.get_from_keys(config, ["forward_dtype"])
        forward_method = cls.get_from_keys(config, ["forward_method"])
        if forward_method in {"scalesweep_mse", "scalesweep_wmse"}:
            expected_lower = -8 if forward_method == "scalesweep_wmse" else -3
            if (config.get("scalesweep_base_rounding", "floor") != "floor"
                    or config.get("scalesweep_lower_bound", expected_lower) != expected_lower
                    or config.get("scalesweep_upper_bound", 7) != 7):
                raise ValueError("Checkpoint ScaleSweep search settings do not match this runtime.")
        lss_identity_transform = config.get("lss_identity_transform", False)
        pseudoquantization = cls.get_from_keys(config, ["pseudoquantization"])
        modules_to_not_convert = cls.get_from_keys(config, ["modules_to_not_convert"])
        return cls(
            hadamard_group_size,
            forward_dtype,
            forward_method,
            lss_identity_transform,
            pseudoquantization,
            modules_to_not_convert,
            scalesweep_identity_transform=config.get("scalesweep_identity_transform", False),
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[LinearMethodBase]:
        if self.modules_to_not_convert is not None and any(
            prefix.endswith(module) for module in self.modules_to_not_convert
        ):
            return UnquantizedLinearMethod()

        if isinstance(layer, LinearBase):
            return FPQuantLinearMethod(self)
        return None


class FPQuantLinearMethod(LinearMethodBase):
    """Linear method for FPQuant.

    Args:
        quant_config: The FPQuant quantization config.
    """

    def __init__(self, quant_config: FPQuantConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del output_size  # Unused.
        del input_size  # Unused.

        if params_dtype != torch.bfloat16:
            raise ValueError("Only bfloat16 is currently supported by FPQuant")
        if self.quant_config.forward_method.startswith("scalesweep_"):
            from vllm.distributed import get_tensor_model_parallel_world_size
            if get_tensor_model_parallel_world_size() != 1:
                raise ValueError("ScaleSweep FPQuant currently supports tensor_parallel_size=1.")
        if input_size_per_partition % self.quant_config.hadamard_group_size != 0:  # noqa: E501
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size. Or other skill issues."
            )

        assert self.quant_config.forward_dtype in ["mxfp4", "nvfp4"], (
            "Only mxfp4 and nvfp4 are supported for now"
        )
        if self.quant_config.forward_dtype == "mxfp4":
            group_size = 32
        elif self.quant_config.forward_dtype == "nvfp4":
            group_size = 16
        else:
            raise ValueError(
                f"Unsupported forward_dtype: {self.quant_config.forward_dtype}"
            )

        qweight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "packed_dim": 1,
                "pack_factor": 2,
            }
            | extra_weight_attrs,
        )
        layer.register_parameter("qweight", qweight)

        scales = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition // group_size,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            scales,
            {
                "input_dim": 1,
                "output_dim": 0,
                "packed_dim": 1,
                "pack_factor": group_size,
            }
            | extra_weight_attrs,
        )
        layer.register_parameter("scales", scales)

        weight_global_scale = Parameter(
            torch.empty(1, dtype=torch.float32),
            requires_grad=False,
        )
        set_weight_attrs(
            weight_global_scale, {"ignore_warning": True} | extra_weight_attrs
        )
        layer.register_parameter("weight_global_scale", weight_global_scale)

        act_global_scale = Parameter(
            torch.empty(1, dtype=torch.float32),
            requires_grad=False,
        )
        set_weight_attrs(
            act_global_scale, {"ignore_warning": True} | extra_weight_attrs
        )
        layer.register_parameter("act_global_scale", act_global_scale)
        if self.quant_config.forward_method == "scalesweep_wmse":
            importance = Parameter(
                torch.full((input_size_per_partition,), float("nan"), dtype=torch.float32),
                requires_grad=False,
            )
            set_weight_attrs(importance, {
                **extra_weight_attrs, "weight_loader": _load_scalesweep_importance,
            })
            layer.register_parameter("act_importance", importance)

        forward_hadamard_matrix = Parameter(
            torch.empty(
                self.quant_config.hadamard_group_size,
                self.quant_config.hadamard_group_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            forward_hadamard_matrix, {"ignore_warning": True} | extra_weight_attrs
        )
        layer.register_parameter("forward_hadamard_matrix", forward_hadamard_matrix)

        backward_hadamard_matrix = Parameter(
            torch.empty(
                self.quant_config.hadamard_group_size,
                self.quant_config.hadamard_group_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            backward_hadamard_matrix, {"ignore_warning": True} | extra_weight_attrs
        )
        layer.register_parameter("backward_hadamard_matrix", backward_hadamard_matrix)

        # Optional Ours channel transform.  Normal FPQuant checkpoints omit
        # these tensors, so identity/ones are safe defaults.
        input_perm = Parameter(
            torch.arange(input_size_per_partition, dtype=torch.long),
            requires_grad=False,
        )
        set_weight_attrs(
            input_perm, {"ignore_warning": True} | extra_weight_attrs
        )
        layer.register_parameter("input_perm", input_perm)

        input_rescale = Parameter(
            torch.ones(input_size_per_partition, dtype=params_dtype),
            requires_grad=False,
        )
        set_weight_attrs(
            input_rescale, {"ignore_warning": True} | extra_weight_attrs
        )
        layer.register_parameter("input_rescale", input_rescale)
        # Set to True by Qwen2Model.load_weights only when a checkpoint
        # actually contains input_perm/input_rescale.
        layer.input_transform_enabled = False

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.quant_config.forward_method.startswith("scalesweep_"):
            for name in ("weight_global_scale", "act_global_scale"):
                scale = getattr(layer, name)
                if not torch.isfinite(scale).all() or not (scale > 0).all():
                    raise ValueError(f"Invalid ScaleSweep checkpoint {name}.")
        if self.quant_config.forward_method == "scalesweep_wmse":
            importance = layer.act_importance
            if (not getattr(importance, "_scalesweep_loaded", False)
                    or not torch.isfinite(importance).all() or (importance < 0).any()):
                raise ValueError("ScaleSweep-WMSE checkpoint is missing valid act_importance.")
        layer._nada_fast_enabled = False
        if (os.getenv("VLLM_FPQUANT_NADA_FAST", "0") == "1"
                and self.quant_config.forward_dtype == "nvfp4"
                and self.quant_config.forward_method == "lss"
                and self.quant_config.lss_identity_transform):
            from vllm.distributed import get_tensor_model_parallel_world_size
            if get_tensor_model_parallel_world_size() != 1:
                raise ValueError("NADA fast inference currently supports tensor parallel size 1.")
            if not hasattr(torch.ops._C, "nvfp4_lss_quant_permute_scale_cutlass"):
                raise RuntimeError("The compiled vLLM extension needs the fused NADA CUTLASS-layout operator.")
            if layer.qweight.shape[1] * 2 % 64 or layer.qweight.shape[0] % 32:
                raise ValueError("NADA fast inference requires input width divisible by 64 and output width by 32.")
            groups = layer.qweight.shape[1] // 8
            layer.register_buffer(
                "_nada_fast_scales",
                to_blocked(layer.scales.view(torch.float8_e4m3fn), backend="triton")
                .view(torch.float8_e4m3fn).reshape(-1, groups).contiguous(),
                persistent=False,
            )
            layer.register_buffer(
                "_nada_fast_perm", layer.input_perm.to(torch.int32).contiguous(),
                persistent=False,
            )
            layer.register_buffer(
                "_nada_fast_alpha", 1.0 / (layer.weight_global_scale * layer.act_global_scale),
                persistent=False,
            )
            layer._nada_fast_enabled = True

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if (getattr(layer, "_nada_fast_enabled", False)
                and os.getenv("VLLM_FPQUANT_TORCH_REFERENCE", "0") != "1"):
            return torch.ops.vllm.nada_fast_forward(
                x, layer.qweight, layer._nada_fast_scales,
                layer._nada_fast_alpha, layer._nada_fast_perm,
                layer.input_rescale, layer.act_global_scale, bias,
            )
        # Set once by the model loader after checkpoint loading. Keep this
        # hot path free of tensor-value tests so torch.compile/CUDA Graph can
        # capture both legacy and Ours checkpoints.
        use_fused_lss_input_transform = (
            layer.input_transform_enabled
            and self.quant_config.forward_dtype == "nvfp4"
            and self.quant_config.forward_method == "lss"
            and self.quant_config.lss_identity_transform
            and os.getenv("VLLM_FPQUANT_TORCH_REFERENCE", "0") != "1"
        )
        if layer.input_transform_enabled and not use_fused_lss_input_transform:
            x = permute_and_scale_cols(
                x, layer.input_perm, layer.input_rescale
            )
        if os.getenv("VLLM_FPQUANT_TORCH_REFERENCE", "0") == "1":
            if not (
                self.quant_config.forward_dtype == "nvfp4"
                and self.quant_config.forward_method == "lss"
                and self.quant_config.lss_identity_transform
            ):
                raise ValueError(
                    "VLLM_FPQUANT_TORCH_REFERENCE supports only NVFP4 LSS "
                    "with an identity transform"
                )
            return _nvfp4_lss_torch_reference(
                x,
                layer.qweight,
                layer.scales,
                layer.weight_global_scale,
                layer.act_global_scale,
                bias,
                (
                    layer.output_partition_sizes
                    if os.getenv(
                        "VLLM_FPQUANT_TORCH_REFERENCE_SPLIT", "0"
                    ) == "1"
                    else None
                ),
            )
        return quantized_forward(
            x,
            layer.qweight,
            layer.scales,
            layer.weight_global_scale,
            layer.act_global_scale,
            bias,
            layer.forward_hadamard_matrix,
            self.quant_config.forward_method,
            self.quant_config.forward_dtype,
            layer.input_perm if use_fused_lss_input_transform else None,
            layer.input_rescale if use_fused_lss_input_transform else None,
            getattr(layer, "act_importance", None),
        )


def _cast_to_fp4_values(x: torch.Tensor) -> torch.Tensor:
    """Round to E2M1 values, matching the export-time pseudoquantizer."""
    sign = torch.sign(x)
    magnitude = x.abs()
    rounded = torch.where(
        magnitude > 5.0,
        6.0,
        torch.where(
            magnitude >= 3.5,
            4.0,
            torch.where(
                magnitude >= 1.75,
                torch.round(magnitude),
                torch.round(magnitude * 2.0) * 0.5,
            ),
        ),
    )
    return rounded * sign


def _nvfp4_lss_quantize_reference(
    x: torch.Tensor, global_scale: torch.Tensor
) -> torch.Tensor:
    original_shape = x.shape
    grouped = x.reshape(-1, x.shape[-1] // 16, 16)
    abs_x = grouped.abs()
    absmax_scale = abs_x.amax(dim=-1, keepdim=True) / 6
    safe_absmax_scale = torch.where(
        absmax_scale == 0, torch.ones_like(absmax_scale), absmax_scale
    )
    grid = _cast_to_fp4_values(abs_x / safe_absmax_scale).abs()
    numerator = (abs_x * grid).sum(dim=-1, keepdim=True)
    denominator = (grid * grid).sum(dim=-1, keepdim=True)
    safe_denominator = torch.where(
        denominator == 0, torch.ones_like(denominator), denominator
    )
    scale = torch.where(
        denominator == 0, absmax_scale, numerator / safe_denominator
    )
    scale = (
        (scale * global_scale)
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
        .to(torch.float32)
        .div(global_scale)
        .to(x.dtype)
    )
    safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    quantized = _cast_to_fp4_values(grouped / safe_scale) * scale
    return quantized.reshape(original_shape)


def _nvfp4_lss_torch_reference(
    x: torch.Tensor,
    qweight: torch.Tensor,
    weight_scales: torch.Tensor,
    weight_global_scale: torch.Tensor,
    act_global_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    output_splits: Optional[list[int]] = None,
) -> torch.Tensor:
    """Slow diagnostic path: decode NVFP4 weights and use BF16 F.linear."""
    fp4_grid = torch.tensor(
        [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
         0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=qweight.device,
    )
    packing_perm = torch.tensor(
        [15, 14, 13, 12, 11, 10, 9, 8, 0, 1, 2, 3, 4, 5, 6, 7],
        dtype=torch.long,
        device=qweight.device,
    )
    codebook = torch.empty_like(fp4_grid)
    codebook[packing_perm] = fp4_grid

    output_features = qweight.shape[0]
    input_features = qweight.shape[1] * 2
    indices = torch.stack((qweight & 0xF, qweight >> 4), dim=-1)
    decoded_weight = codebook[
        indices.reshape(output_features, input_features).long()
    ]
    decoded_scales = (
        weight_scales.view(torch.float8_e4m3fn).float()
        / weight_global_scale
    )
    decoded_weight = (
        decoded_weight * decoded_scales.repeat_interleave(16, dim=1)
    ).to(torch.bfloat16)
    quantized_x = _nvfp4_lss_quantize_reference(x, act_global_scale)
    if output_splits is not None and len(output_splits) > 1:
        weights = decoded_weight.split(output_splits, dim=0)
        biases = (
            bias.split(output_splits, dim=0)
            if bias is not None
            else [None] * len(output_splits)
        )
        return torch.cat(
            [
                F.linear(quantized_x, weight, split_bias)
                for weight, split_bias in zip(weights, biases)
            ],
            dim=-1,
        )
    return F.linear(quantized_x, decoded_weight, bias)


def ceil_div(a, b):
    return (a + b - 1) // b


def fused_quantize_mx(
    x_flat: torch.Tensor, hadamard_matrix: torch.Tensor, forward_method: str
) -> tuple[torch.Tensor, torch.Tensor]:
    return fusedQuantizeMx(x_flat, hadamard_matrix, method=forward_method)


def fused_quantize_mx_fake(x_flat, hadamard_matrix, forward_method):
    rows, cols = x_flat.size(0), x_flat.size(1) // 32
    padded_rows = ((rows + 128 - 1) // 128) * 128
    padded_cols = ((cols + 4 - 1) // 4) * 4

    xh_e2m1 = torch.empty(
        x_flat.size(0), x_flat.size(1) // 2, dtype=torch.uint8, device=x_flat.device
    )
    xh_e8m0 = torch.empty(
        padded_rows, padded_cols, dtype=torch.float8_e8m0fnu, device=x_flat.device
    )

    return xh_e2m1, xh_e8m0


direct_register_custom_op(
    op_name="fused_quantize_mx",
    op_func=fused_quantize_mx,
    mutates_args=[],
    fake_impl=fused_quantize_mx_fake,
    dispatch_key=current_platform.dispatch_key,
)


def matmul_mxf4_bf16(
    x: torch.Tensor,
    w: torch.Tensor,
    xs: torch.Tensor,
    ws: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    return matmul_mxf4_bf16_tn(
        x,
        w,
        to_blocked(xs, backend="triton").view(torch.float8_e8m0fnu),
        to_blocked(ws, backend="triton").view(torch.float8_e8m0fnu),
        alpha,
    )


def matmul_mxf4_bf16_fake(x, w, xs, ws, alpha):
    return torch.empty(*x.shape[:-1], w.shape[0], dtype=torch.bfloat16, device=x.device)


direct_register_custom_op(
    op_name="matmul_mxf4_bf16",
    op_func=matmul_mxf4_bf16,
    mutates_args=[],
    fake_impl=matmul_mxf4_bf16_fake,
    dispatch_key=current_platform.dispatch_key,
)


def fused_quantize_nv(
    x_flat: torch.Tensor, hadamard_matrix: torch.Tensor, global_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return fusedQuantizeNv(x_flat, hadamard_matrix, global_scale)


def fused_quantize_nv_fake(x_flat, hadamard_matrix, global_scale):
    rows, cols = x_flat.size(0), x_flat.size(1) // 16
    padded_rows = ((rows + 128 - 1) // 128) * 128
    padded_cols = ((cols + 4 - 1) // 4) * 4

    xh_e2m1 = torch.empty(
        x_flat.size(0), x_flat.size(1) // 2, dtype=torch.uint8, device=x_flat.device
    )
    xh_e8m0 = torch.empty(
        padded_rows, padded_cols, dtype=torch.float8_e4m3fn, device=x_flat.device
    )

    return xh_e2m1, xh_e8m0


direct_register_custom_op(
    op_name="fused_quantize_nv",
    op_func=fused_quantize_nv,
    mutates_args=[],
    fake_impl=fused_quantize_nv_fake,
    dispatch_key=current_platform.dispatch_key,
)


def nvfp4_lss_quant_identity_fake(x_flat, global_scale):
    rows, groups = x_flat.size(0), x_flat.size(1) // 16
    return (
        torch.empty(
            rows, x_flat.size(1) // 2, dtype=torch.uint8, device=x_flat.device
        ),
        torch.empty(
            ((rows + 127) // 128) * 128,
            ((groups + 3) // 4) * 4,
            dtype=torch.float8_e4m3fn,
            device=x_flat.device,
        ),
    )


def nvfp4_lss_quant_permute_scale_fake(x_flat, perm, scale, global_scale):
    return nvfp4_lss_quant_identity_fake(x_flat, global_scale)


direct_register_custom_op(
    op_name="nvfp4_lss_quant_identity",
    op_func=nvfp4_lss_quant_identity,
    mutates_args=[],
    fake_impl=nvfp4_lss_quant_identity_fake,
    dispatch_key=current_platform.dispatch_key,
)

direct_register_custom_op(
    op_name="nvfp4_scalesweep_quant",
    op_func=scalesweep_quantize,
    mutates_args=[],
    fake_impl=scalesweep_quantize_fake,
    dispatch_key=current_platform.dispatch_key,
)


direct_register_custom_op(
    op_name="nvfp4_lss_quant_permute_scale",
    op_func=nvfp4_lss_quant_permute_scale,
    mutates_args=[],
    fake_impl=nvfp4_lss_quant_permute_scale_fake,
    dispatch_key=current_platform.dispatch_key,
)


def matmul_nvf4_bf16(
    x: torch.Tensor,
    w: torch.Tensor,
    xs: torch.Tensor,
    ws: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    return cutlass_scaled_fp4_mm(
        x,
        w,
        to_blocked(xs, backend="triton")
        .view(torch.float8_e4m3fn)
        .view(-1, x.shape[1] // 8),  # *2//16
        to_blocked(ws, backend="triton")
        .view(torch.float8_e4m3fn)
        .view(-1, x.shape[1] // 8),
        alpha,
        torch.bfloat16,
    )


def matmul_nvf4_bf16_fake(x, w, xs, ws, alpha):
    return torch.empty(*x.shape[:-1], w.shape[0], dtype=torch.bfloat16, device=x.device)


direct_register_custom_op(
    op_name="matmul_nvf4_bf16",
    op_func=matmul_nvf4_bf16,
    mutates_args=[],
    fake_impl=matmul_nvf4_bf16_fake,
    dispatch_key=current_platform.dispatch_key,
)


def nada_fast_forward(
    x: torch.Tensor, weight: torch.Tensor, blocked_weight_scales: torch.Tensor,
    alpha: torch.Tensor, perm: torch.Tensor, rescale: torch.Tensor,
    act_global_scale: torch.Tensor, bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """Fused NADA activation quantization directly into the CUTLASS layout."""
    flat = x.contiguous().flatten(end_dim=-2)
    values, scales = nvfp4_lss_quant_permute_scale_cutlass(
        flat, perm, rescale, act_global_scale
    )
    # Pass the logical row count to GEMM. The backing buffers remain padded
    # for the block-scaled layout, including single-token decode.
    output = cutlass_scaled_fp4_mm(
        values[:flat.shape[0]], weight,
        scales.view(-1, flat.shape[1] // 16),
        blocked_weight_scales, alpha, torch.bfloat16,
    )
    output = output.view(*x.shape[:-1], weight.shape[0])
    return output + bias if bias is not None else output


def nada_fast_forward_fake(
    x: torch.Tensor, weight: torch.Tensor, blocked_weight_scales: torch.Tensor,
    alpha: torch.Tensor, perm: torch.Tensor, rescale: torch.Tensor,
    act_global_scale: torch.Tensor, bias: Optional[torch.Tensor],
) -> torch.Tensor:
    return torch.empty((*x.shape[:-1], weight.shape[0]),
                       device=x.device, dtype=torch.bfloat16)


direct_register_custom_op(
    op_name="nada_fast_forward", op_func=nada_fast_forward,
    mutates_args=[], fake_impl=nada_fast_forward_fake,
    dispatch_key=current_platform.dispatch_key,
)


def quantized_forward(
    x: torch.Tensor,
    qweight: torch.Tensor,
    weight_scales: torch.Tensor,
    weight_global_scale: torch.Tensor,
    act_global_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    forward_hadamard_matrix: torch.Tensor,
    forward_method: str,
    forward_dtype: str,
    input_perm: Optional[torch.Tensor] = None,
    input_rescale: Optional[torch.Tensor] = None,
    act_importance: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x_flat = x.contiguous().flatten(end_dim=-2)

    if forward_dtype == "mxfp4":
        x_flat_q, x_flat_scales = torch.ops.vllm.fused_quantize_mx(
            x_flat, forward_hadamard_matrix, forward_method
        )
        y = torch.ops.vllm.matmul_mxf4_bf16(
            x_flat_q,
            qweight,
            x_flat_scales,
            weight_scales,
            1 / (weight_global_scale * act_global_scale),
        )
    elif forward_dtype == "nvfp4":
        if forward_method == "abs_max":
            x_flat_q, x_flat_scales = torch.ops.vllm.fused_quantize_nv(
                x_flat, forward_hadamard_matrix, act_global_scale
            )
        elif forward_method == "lss":
            if input_perm is None:
                x_flat_q, x_flat_scales = torch.ops.vllm.nvfp4_lss_quant_identity(
                    x_flat, act_global_scale
                )
            else:
                assert input_rescale is not None
                x_flat_q, x_flat_scales = (
                    torch.ops.vllm.nvfp4_lss_quant_permute_scale(
                        x_flat, input_perm, input_rescale, act_global_scale
                    )
                )
        elif forward_method in {"scalesweep_mse", "scalesweep_wmse"}:
            x_flat_q, x_flat_scales = torch.ops.vllm.nvfp4_scalesweep_quant(
                x_flat, act_global_scale, act_importance,
                forward_method == "scalesweep_wmse",
            )
        else:
            raise ValueError(f"Unsupported NVFP4 forward_method: {forward_method}")
        y = torch.ops.vllm.matmul_nvf4_bf16(
            x_flat_q,
            qweight,
            x_flat_scales,
            weight_scales,
            1 / (weight_global_scale * act_global_scale),
        )
    else:
        raise ValueError(f"Unsupported forward_dtype: {forward_dtype}")

    y = y.view(*x.shape[:-1], y.shape[-1])
    if bias is not None:
        y += bias

    return y
