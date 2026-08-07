"""General-purpose utilities for hallucination-reduced-vqa.

Contains seed management, retry logic, and miscellaneous helpers shared
across all modules.

Example:
    >>> from src.utils.helpers import set_seed, retry_on_failure
    >>> set_seed(42)
"""

import logging
import random
import time
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

import numpy as np

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# ============================================================
# Reproducibility
# ============================================================


def set_seed(seed: int = 42) -> None:
    """Set random seeds for full reproducibility across all libraries.

    Sets seeds for Python's ``random``, NumPy, and PyTorch (CPU + CUDA).
    Call this once at the start of each script or training run.

    Args:
        seed: Integer seed value.  SRS default is 42.

    Example:
        >>> set_seed(42)
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.debug(f"PyTorch seed set to {seed}")
    except ImportError:
        pass
    logger.debug(f"Random seeds set: seed={seed}")


# ============================================================
# Retry decorator
# ============================================================


def retry_on_failure(
    max_retries: int = 3,
    delay: float = 2.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable[[F], F]:
    """Decorator: retry a function on failure with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        delay: Initial delay in seconds between retries.
        backoff: Multiplicative factor applied to ``delay`` on each retry.
        exceptions: Tuple of exception types to catch and retry.

    Returns:
        Decorated function that retries on the specified exceptions.

    Example:
        >>> @retry_on_failure(max_retries=3, delay=1.0)
        ... def download_file(url: str) -> bytes:
        ...     ...
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            wait = delay
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    if attempt == max_retries:
                        logger.error(
                            f"{func.__name__} failed after {max_retries} attempts: {exc}"
                        )
                        raise
                    logger.warning(
                        f"{func.__name__} attempt {attempt}/{max_retries} failed: {exc}. "
                        f"Retrying in {wait:.1f}s..."
                    )
                    time.sleep(wait)
                    wait *= backoff

        return wrapper  # type: ignore[return-value]

    return decorator


# ============================================================
# Miscellaneous
# ============================================================


def format_bytes(n_bytes: int) -> str:
    """Format a byte count as a human-readable string.

    Args:
        n_bytes: Number of bytes.

    Returns:
        Human-readable string, e.g. ``"1.23 GB"``.

    Example:
        >>> format_bytes(1_500_000_000)
        '1.40 GB'
    """
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024.0:
            return f"{n_bytes:.2f} {unit}"
        n_bytes /= 1024.0
    return f"{n_bytes:.2f} PB"


def clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` between ``low`` and ``high``.

    Args:
        value: Input value.
        low: Lower bound.
        high: Upper bound.

    Returns:
        Clamped value.
    """
    return max(low, min(high, value))
