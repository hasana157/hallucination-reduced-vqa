"""Standalone script to execute Module 2 QLoRA Fine-Tuning Pipeline.

Usage:
    python scripts/train.py --data_root ./data --output_dir checkpoints/lora_weights --num_epochs 2
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_data_root
from src.data import VQAv2Loader
from src.models import LoRAAdapter, QwenVLQuantizer
from src.training import QLoRATrainer, TrainingConfig, VQADataset
from src.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Run QLoRA Fine-Tuning on VQAv2 dataset.")
    parser.add_argument("--data_root", type=str, default=None, help="Root directory for dataset storage")
    parser.add_argument("--output_dir", type=str, default="checkpoints/lora_weights", help="Directory to save LoRA checkpoints")
    parser.add_argument("--num_epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=2, help="Per device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Number of warmup steps")
    parser.add_argument("--save_steps", type=int, default=200, help="Checkpoint save frequency in steps")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluation frequency in steps")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Maximum number of checkpoints to keep")
    parser.add_argument("--lora_rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha scaling")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Checkpoint folder path to resume from")
    args = parser.parse_args()

    setup_logging("logs/train.log")

    if args.data_root:
        import os
        os.environ["DATA_ROOT"] = args.data_root

    data_root = get_data_root()

    print("==================================================")
    print("QLoRA Fine-Tuning Pipeline (Module 2)")
    print(f"Data Root: {data_root}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Epochs: {args.num_epochs} | LR: {args.learning_rate} | Batch Size: {args.batch_size}")
    print("==================================================")

    # 1. Load Datasets
    print("\n[1/4] Loading VQAv2 Dataset...")
    vqa_loader = VQAv2Loader(data_root=data_root)
    train_dict = vqa_loader.load(split="train")
    val_dict = vqa_loader.load(split="val")

    train_dataset = VQADataset(train_dict)
    val_dataset = VQADataset(val_dict) if val_dict else None

    print(f"  ✓ Loaded Train items: {len(train_dataset)}, Val items: {len(val_dataset) if val_dataset else 0}")

    # 2. Load 4-bit Quantized Model
    print("\n[2/4] Loading Qwen2-VL-2B-Instruct in 4-bit Quantization...")
    quantizer = QwenVLQuantizer()
    model, processor = quantizer.load("Qwen/Qwen2-VL-2B-Instruct")

    # 3. Apply LoRA Adapter
    print("\n[3/4] Applying LoRA Adapters...")
    model = LoRAAdapter.apply(model, r=args.lora_rank, alpha=args.lora_alpha)

    # 4. Configure & Train
    print("\n[4/4] Starting Training...")
    cfg = TrainingConfig(
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        output_dir=args.output_dir,
        lora_r=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )

    trainer = QLoRATrainer(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=cfg,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    print("\n==================================================")
    print("QLoRA Fine-Tuning Complete!")
    print("==================================================")


if __name__ == "__main__":
    main()
