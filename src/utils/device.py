"""Device management utilities for hallucination-reduced-vqa.

Handles PyTorch device selection (CUDA / CPU), GPU memory inspection,
and Colab hardware check.

Example:
    >>> from src.utils.device import get_device, print_gpu_memory
    >>> device = get_device()
    >>> print_gpu_memory()
"""

import logging
from typing import Dict, Tuple

import torch

logger = logging.getLogger(__name__)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return available PyTorch device.

    Args:
        prefer_cuda: If True, uses CUDA if available.

    Returns:
        torch.device ("cuda" or "cpu").
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Using CUDA device: {gpu_name}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    return device


def get_gpu_memory() -> Dict[str, float]:
    """Return current CUDA memory stats in Gigabytes (GB).

    Returns:
        Dict with keys: "allocated", "reserved", "max_allocated", "total".
    """
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0, "max_allocated": 0.0, "total": 0.0}

    device_idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_idx)
    total = props.total_memory / (1024**3)
    allocated = torch.cuda.memory_allocated(device_idx) / (1024**3)
    reserved = torch.cuda.memory_reserved(device_idx) / (1024**3)
    max_allocated = torch.cuda.max_memory_allocated(device_idx) / (1024**3)

    return {
        "allocated": round(allocated, 3),
        "reserved": round(reserved, 3),
        "max_allocated": round(max_allocated, 3),
        "total": round(total, 3),
    }


def print_gpu_memory() -> None:
    """Log formatted CUDA memory status."""
    mem = get_gpu_memory()
    if mem["total"] > 0:
        logger.info(
            f"GPU Memory — Allocated: {mem['allocated']} GB | "
            f"Reserved: {mem['reserved']} GB | "
            f"Peak: {mem['max_allocated']} GB / {mem['total']} GB"
        )
    else:
        logger.info("GPU Memory — No CUDA GPU available")


def is_colab() -> bool:
    """Check if execution environment is Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False
