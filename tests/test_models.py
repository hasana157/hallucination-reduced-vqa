"""Unit tests for Model Components."""

import pytest
from src.models import LoRAAdapter


def test_lora_config_generation():
    """LoRAAdapter generates PEFT LoraConfig with expected params."""
    try:
        import peft
    except ImportError:
        pytest.skip("peft package not installed in environment")

    config = LoRAAdapter.get_config(r=16, alpha=32)
    assert config.r == 16
    assert config.lora_alpha == 32
    assert "q_proj" in config.target_modules
    assert "v_proj" in config.target_modules
