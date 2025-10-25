"""
Quantization utilities using torch tao for model quantization before vllm deployment.

This module provides functionality to quantize models using torch tao's quantization
techniques, optimizing them for inference while maintaining quality.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from torchao.quantization import (
    Int8WeightOnlyConfig,
    Int4WeightOnlyConfig,
    quantize_,
)

logger = logging.getLogger(__name__)


class QuantizationConfig:
    """Configuration for model quantization."""

    def __init__(
        self,
        method: str = "int8",
        group_size: Optional[int] = None,
        inner_k_tokens: Optional[int] = None,
        enable_activation_quantization: bool = False,
    ):
        """
        Initialize quantization config.

        Args:
            method: Quantization method - "int8", "int4", or "nf4"
            group_size: Group size for quantization (used for int4). None means per-channel.
            inner_k_tokens: Inner K tokens for activation-aware quantization.
            enable_activation_quantization: Whether to quantize activations.
        """
        self.method = method
        self.group_size = group_size
        self.inner_k_tokens = inner_k_tokens
        self.enable_activation_quantization = enable_activation_quantization

    def get_torch_ao_config(self):
        """Get the corresponding torch tao quantization config."""
        if self.method == "int8":
            return Int8WeightOnlyConfig()
        elif self.method == "int4":
            return Int4WeightOnlyConfig(
                group_size=self.group_size,
                inner_k_tokens=self.inner_k_tokens,
            )
        else:
            raise ValueError(f"Unknown quantization method: {self.method}")


def quantize_model(
    model: nn.Module,
    config: QuantizationConfig | dict | None = None,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Quantize a model using torch tao quantization.

    Args:
        model: The model to quantize.
        config: QuantizationConfig object or dict with quantization parameters.
                Defaults to Int8WeightOnlyConfig if None.
        device: Device to move the model to after quantization.

    Returns:
        Quantized model.

    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
        >>> config = QuantizationConfig(method="int8")
        >>> quantized_model = quantize_model(model, config)
    """
    if config is None:
        config = QuantizationConfig(method="int8")
    elif isinstance(config, dict):
        config = QuantizationConfig(**config)

    logger.info(f"Starting quantization with method: {config.method}")

    torch_ao_config = config.get_torch_ao_config()

    # Quantize the model in-place
    quantize_(model, torch_ao_config)

    logger.info(f"Quantization completed successfully with {config.method}")

    if device is not None:
        model = model.to(device)
        logger.info(f"Model moved to device: {device}")

    return model


def quantize_and_save(
    model: nn.Module,
    output_path: str,
    config: QuantizationConfig | dict | None = None,
    device: Optional[torch.device] = None,
) -> None:
    """
    Quantize a model and save it.

    Args:
        model: The model to quantize.
        output_path: Path to save the quantized model.
        config: Quantization configuration.
        device: Device to move model to after quantization.
    """
    quantized_model = quantize_model(model, config, device)
    torch.save(quantized_model.state_dict(), output_path)
    logger.info(f"Quantized model saved to {output_path}")


def get_quantization_config(
    method: str = "int8",
    **kwargs,
) -> QuantizationConfig:
    """
    Factory function to create quantization config.

    Args:
        method: Quantization method ("int8", "int4", "nf4").
        **kwargs: Additional arguments to pass to QuantizationConfig.

    Returns:
        QuantizationConfig object.
    """
    return QuantizationConfig(method=method, **kwargs)
