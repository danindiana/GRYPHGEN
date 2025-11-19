"""Model loading and optimization utilities."""

import logging
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model loading."""

    model_name: str
    device: str = "cuda:0"
    dtype: torch.dtype = torch.float16
    max_memory: Optional[Dict[Union[int, str], str]] = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_flash_attention: bool = True
    trust_remote_code: bool = False


class ModelOptimizer:
    """
    Model optimization utilities for RTX 4080.

    Provides methods to load and optimize transformer models
    for efficient inference on NVIDIA RTX 4080.
    """

    @staticmethod
    def load_model_optimized(
        model_name: str,
        config: Optional[ModelConfig] = None,
        use_safetensors: bool = True,
    ) -> tuple:
        """
        Load a model with RTX 4080 optimizations.

        Args:
            model_name: HuggingFace model name or path
            config: Model configuration
            use_safetensors: Use safetensors format for faster loading

        Returns:
            Tuple of (model, tokenizer)
        """
        if config is None:
            config = ModelConfig(model_name=model_name)

        logger.info(f"Loading model: {model_name}")
        logger.info(f"Device: {config.device}, dtype: {config.dtype}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=config.trust_remote_code,
        )

        # Prepare model loading kwargs
        model_kwargs = {
            "pretrained_model_name_or_path": model_name,
            "torch_dtype": config.dtype,
            "trust_remote_code": config.trust_remote_code,
            "use_safetensors": use_safetensors,
        }

        # Add quantization if requested
        if config.load_in_8bit:
            model_kwargs["load_in_8bit"] = True
            logger.info("Loading model in 8-bit quantization")
        elif config.load_in_4bit:
            model_kwargs["load_in_4bit"] = True
            logger.info("Loading model in 4-bit quantization")

        # Add max memory if specified
        if config.max_memory:
            model_kwargs["max_memory"] = config.max_memory

        # Add flash attention if supported
        if config.use_flash_attention:
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Using Flash Attention 2")
            except Exception as e:
                logger.warning(f"Flash Attention not available: {e}")

        # Load model
        try:
            model = AutoModel.from_pretrained(**model_kwargs)

            # Move to device if not using device_map
            if not config.load_in_8bit and not config.load_in_4bit:
                model = model.to(config.device)

            # Set to eval mode
            model.eval()

            # Enable optimizations
            if hasattr(torch.cuda, 'amp') and config.dtype == torch.float16:
                logger.info("Model ready for automatic mixed precision")

            logger.info(f"Model loaded successfully: {model_name}")

            return model, tokenizer

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    @staticmethod
    def optimize_inference(model: torch.nn.Module) -> torch.nn.Module:
        """
        Optimize model for inference.

        Args:
            model: PyTorch model

        Returns:
            Optimized model
        """
        # Set to eval mode
        model.eval()

        # Disable gradient computation
        for param in model.parameters():
            param.requires_grad = False

        # Try to compile model (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            try:
                logger.info("Compiling model with torch.compile")
                model = torch.compile(model, mode="max-autotune")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")

        return model

    @staticmethod
    def estimate_memory_usage(
        model_name: str,
        dtype: torch.dtype = torch.float16,
    ) -> float:
        """
        Estimate model memory usage in GB.

        Args:
            model_name: Model name
            dtype: Data type

        Returns:
            Estimated memory in GB
        """
        try:
            config = AutoConfig.from_pretrained(model_name)

            # Rough estimation based on parameters
            if hasattr(config, 'num_parameters'):
                num_params = config.num_parameters
            else:
                # Estimate from hidden size and layers
                num_params = (
                    config.hidden_size * config.hidden_size * config.num_hidden_layers * 12
                )

            # Calculate memory based on dtype
            bytes_per_param = {
                torch.float32: 4,
                torch.float16: 2,
                torch.bfloat16: 2,
                torch.int8: 1,
            }.get(dtype, 4)

            memory_gb = (num_params * bytes_per_param) / (1024 ** 3)

            # Add overhead for activations (rough estimate: 2x parameters)
            total_memory_gb = memory_gb * 3

            logger.info(f"Estimated memory for {model_name}: {total_memory_gb:.2f} GB")

            return total_memory_gb

        except Exception as e:
            logger.warning(f"Could not estimate memory: {e}")
            return 0.0


def load_model_optimized(
    model_name: str,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.float16,
    quantization: Optional[str] = None,
) -> tuple:
    """
    Convenience function to load an optimized model.

    Args:
        model_name: Model name or path
        device: Device to load on
        dtype: Data type
        quantization: Quantization type ("8bit", "4bit", or None)

    Returns:
        Tuple of (model, tokenizer)
    """
    config = ModelConfig(
        model_name=model_name,
        device=device,
        dtype=dtype,
        load_in_8bit=(quantization == "8bit"),
        load_in_4bit=(quantization == "4bit"),
    )

    optimizer = ModelOptimizer()
    model, tokenizer = optimizer.load_model_optimized(model_name, config)

    # Further optimize for inference
    model = optimizer.optimize_inference(model)

    return model, tokenizer


def get_recommended_dtype(compute_capability: tuple) -> torch.dtype:
    """
    Get recommended dtype based on GPU compute capability.

    Args:
        compute_capability: Tuple of (major, minor) version

    Returns:
        Recommended torch dtype
    """
    major, minor = compute_capability

    # Ada Lovelace (RTX 4080) - compute capability 8.9
    # Supports bfloat16 natively
    if major >= 8:
        logger.info("Using bfloat16 (recommended for Ada Lovelace/Ampere)")
        return torch.bfloat16

    # Turing and newer support float16 well
    elif major >= 7:
        logger.info("Using float16 (recommended for Turing/Volta)")
        return torch.float16

    # Older GPUs
    else:
        logger.info("Using float32 (older GPU architecture)")
        return torch.float32
