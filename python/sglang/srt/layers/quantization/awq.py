# SPDX-License-Identifier: Apache-2.0
import logging
from typing import Any, Dict, List, Optional

import torch
from vllm import _custom_ops as vllm_ops
from vllm.model_executor.layers.quantization.awq_triton import awq_dequantize_triton
from sglang.srt.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from sglang.srt.layers.parameter import GroupQuantScaleParameter, PackedvLLMParameter
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.utils import is_cuda

logger = logging.getLogger(__name__)


def is_layer_skipped_awq(prefix: str, modules_to_not_convert: List[str]):
    return any(module_name in prefix for module_name in modules_to_not_convert)

def awq_dequantize_wrapper(qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor) -> torch.Tensor:
    if is_cuda():
        return vllm_ops.awq_dequantize(
                    qweight,
                    scales,
                    qzeros,
                    0,
                    0,
                    0,
                )

    # qweight: (K, N / 8), int32
    # qzeros: (K / group_size, N / 8), int32
    # scales: (K / group_size, N), bfloat16
    # Torch implementation of awq_dequantize
    bitshifts = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], dtype=torch.int32, device=qweight.device) * 4
    qweight_unpacked = (qweight.unsqueeze(-1) >> bitshifts) & 0xF
    qweight_unpacked = qweight_unpacked.flatten(-2)  # (K, N)

    qzeros_unpacked = (qzeros.unsqueeze(-1) >> bitshifts) & 0xF
    qzeros_unpacked = qzeros_unpacked.flatten(-2)  # (K / group_size, N)

    num_groups = qzeros.shape[0]
    qweight_unpacked = qweight_unpacked.unflatten(0, (num_groups, -1))
    qweight = qweight_unpacked - qzeros_unpacked.unsqueeze(1)
    weight = qweight.float() * scales.unsqueeze(1).float()
    weight = weight.flatten(0, 1).to(scales.dtype)
    return weight

class AWQConfig(QuantizationConfig):
    """Config class for AWQ.

    Reference: https://arxiv.org/abs/2306.00978
    """

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        zero_point: bool,
        modules_to_not_convert: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.zero_point = zero_point
        self.modules_to_not_convert = modules_to_not_convert or []

        if self.weight_bits != 4:
            raise ValueError(
                "Currently, only 4-bit weight quantization is supported for "
                f"AWQ, but got {self.weight_bits} bits."
            )
        self.pack_factor = 32 // self.weight_bits

    def __repr__(self) -> str:
        return (
            f"AWQConfig(weight_bits={self.weight_bits}, "
            f"group_size={self.group_size}, "
            f"zero_point={self.zero_point}, "
            f"modules_to_not_convert={self.modules_to_not_convert})"
        )

    def get_scaled_act_names(self) -> List[str]:
        return []

    def get_name(self) -> str:
        return "awq"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        # The AWQ kernel only supports Turing or newer GPUs.
        return 75

    @staticmethod
    def get_config_filenames() -> List[str]:
        return [
            "quant_config.json",  # E.g., casperhansen/vicuna-7b-v1.5-awq
            # E.g., abhinavkulkarni/mosaicml-mpt-7b-instruct-w4-g128-awq
            "quantize_config.json",
        ]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AWQConfig":
        weight_bits = cls.get_from_keys(config, ["w_bit", "bits"])
        group_size = cls.get_from_keys(config, ["q_group_size", "group_size"])
        zero_point = cls.get_from_keys(config, ["zero_point"])
        modules_to_not_convert = cls.get_from_keys_or(
            config, ["modules_to_not_convert"], None
        )
        return cls(weight_bits, group_size, zero_point, modules_to_not_convert)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["LinearMethodBase"]:

        if isinstance(layer, LinearBase):
            if is_layer_skipped_awq(prefix, self.modules_to_not_convert):
                return UnquantizedLinearMethod()
            return AWQLinearMethod(self)
        return None


class AWQLinearMethod(LinearMethodBase):
    """Linear method for AWQ.

    Args:
        quant_config: The AWQ quantization config.
    """

    def __init__(self, quant_config: AWQConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        if input_size_per_partition % self.quant_config.group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size."
            )

        output_size_per_partition = sum(output_partition_sizes)
        if output_size_per_partition % self.quant_config.pack_factor != 0:
            raise ValueError(
                "The output size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size."
            )

        weight_loader = extra_weight_attrs.get("weight_loader")
        qweight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader,
        )

        qzeros = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition // self.quant_config.group_size,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader,
        )

        scales = GroupQuantScaleParameter(
            data=torch.empty(
                input_size_per_partition // self.quant_config.group_size,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=0,
            output_dim=1,
            weight_loader=weight_loader,
        )

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("qzeros", qzeros)
        layer.register_parameter("scales", scales)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.qweight = torch.nn.Parameter(layer.qweight.data, requires_grad=False)
        layer.qzeros = torch.nn.Parameter(layer.qzeros.data, requires_grad=False)
        layer.scales = torch.nn.Parameter(layer.scales.data, requires_grad=False)
        # Warmup for cutlass kernel
        if self.quant_config.group_size % 32 or layer.scales.data.dtype == torch.bfloat16:
            pass
        else:
            qweight = vllm_ops.awq_to_gptq_4bit(layer.qweight)
            layer.qweight = torch.nn.Parameter(qweight, requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # qweight = layer.qweight
        # scales = layer.scales
        # qzeros = layer.qzeros
        # pack_factor = self.quant_config.pack_factor
        # out_shape = x.shape[:-1] + (qweight.shape[-1] * pack_factor,)
        # reshaped_x = x.reshape(-1, x.shape[-1])

        # out = awq_dequantize(qweight, scales, qzeros)
        # out = torch.matmul(reshaped_x, out)

        # if bias is not None:
        #     out.add_(bias)
        # return out.reshape(out_shape)

        qweight = layer.qweight
        scales = layer.scales
        qzeros = layer.qzeros
        pack_factor = self.quant_config.pack_factor
        reshaped_x = x.reshape(-1, x.shape[-1])
        out = torch.empty(0)
        out_shape = ()

        if self.quant_config.group_size % 32 or x.dtype == torch.bfloat16:
            out_shape = (x.shape[:-1] + (qweight.shape[-1] * pack_factor, ))
            out = awq_dequantize_wrapper(qweight, scales, qzeros)
            if reshaped_x.dtype == torch.bfloat16:
                out = out.to(torch.bfloat16)
            out = torch.matmul(reshaped_x, out)
        else:
            num_out_channel = qweight.shape[0]
            out_shape = (x.shape[:-1] + (num_out_channel, ))
            temp_space = torch.empty(0, dtype=torch.float32, device=x.device)
            if reshaped_x.dtype == torch.bfloat16:
                temp_space = torch.zeros(reshaped_x.shape[0], num_out_channel,
                                         dtype=torch.float32, device=x.device)
            out = vllm_ops.awq_gemm(reshaped_x, qweight, qzeros, scales,
                                    pack_factor, temp_space,
                                    True if reshaped_x.dtype == torch.bfloat16 else False)
        if bias is not None:
            out.add_(bias)
        return out.reshape(out_shape)
