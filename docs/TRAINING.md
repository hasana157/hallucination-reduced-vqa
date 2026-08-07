# Module 2 — QLoRA Fine-Tuning Pipeline

## Overview

Module 2 fine-tunes **Qwen2-VL-2B-Instruct** using 4-bit quantization (**BitsAndBytes NF4**) and Low-Rank Adaptation (**PEFT LoRA**).

---

## Key Hyperparameters

| Hyperparameter | Value | Description |
|----------------|-------|-------------|
| Model | `Qwen/Qwen2-VL-2B-Instruct` | Base Vision-Language Model |
| Quantization | 4-bit NF4 | BitsAndBytes double quantization (~2.8GB VRAM) |
| LoRA Rank ($r$) | 16 | Matrix decomposition rank |
| LoRA Alpha ($\alpha$) | 32 | Scaling factor ($\alpha / r = 2.0$) |
| Target Modules | `q_proj`, `v_proj`, `up_proj`, `down_proj`, `gate_proj` | Linear projection targets |
| Learning Rate | `1e-4` | AdamW with linear warmup (500 steps) |
| Batch Size | 4 | Per GPU (effective batch = 4 × 2 grad accum = 8) |
| Epochs | 2 | ~3 GPU hours on Google Colab T4 |
| Checkpoint Size | < 50MB | Independent adapter checkpoint |

---

## Command Line Usage

```bash
# Run QLoRA Training
python scripts/train.py \
  --data_root ./data \
  --output_dir checkpoints/lora_weights \
  --num_epochs 2 \
  --learning_rate 1e-4

# Resume Training from Checkpoint
python scripts/train.py \
  --data_root ./data \
  --output_dir checkpoints/lora_weights \
  --resume_from_checkpoint checkpoints/lora_weights/checkpoint-5000
```

---

## Python API

```python
from src.models import QwenVLQuantizer, LoRAAdapter
from src.training import QLoRATrainer, TrainingConfig, VQADataset

# 1. Load 4-bit base model
quantizer = QwenVLQuantizer()
model, processor = quantizer.load("Qwen/Qwen2-VL-2B-Instruct")

# 2. Attach LoRA adapters
model = LoRAAdapter.apply(model, r=16, alpha=32)

# 3. Train
trainer = QLoRATrainer(
    model=model,
    processor=processor,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    config=TrainingConfig(num_epochs=2),
)
trainer.train()
```
