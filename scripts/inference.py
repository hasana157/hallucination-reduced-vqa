#!/usr/bin/env python3
"""Inference CLI: run batch inference for configurations A, B, or C.

Runs InferencePipeline over a dataset split and saves per-query results to JSON.

Usage:
    python scripts/inference.py --mode C --data_root ./data \\
        --lora_path checkpoints/lora_weights \\
        --output_file results/inference_C.json --limit 10

    python scripts/inference.py --mode A --limit 50 --output_file results/inference_A.json

Required Arguments:
    --mode          {A, B, C}  — Inference configuration

Optional Arguments:
    --data_root     Root data directory (default: ./data)
    --lora_path     Path to LoRA checkpoint (default: checkpoints/lora_weights)
    --faiss_index   Path to FAISS index directory (default: data/faiss_index)
    --output_file   Output JSON file path (default: results/inference_{mode}.json)
    --limit         Max number of items to process (default: all)
    --dataset       Dataset to use: vqav2 or pope (default: vqav2)
    --pope_mode     POPE subset: random, popular, adversarial (default: random)
    --config_file   Path to inference_config.yaml (default: config/inference_config.yaml)
    --seed          Random seed (default: 42)
    --checkpoint    Path for batch checkpoint/resume file
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference import InferenceConfig, InferencePipeline, InferenceResult
from src.data import FAISSRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("scripts.inference")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run batch VQA inference (modes A/B/C)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--mode",
        choices=["A", "B", "C"],
        required=True,
        help="Inference configuration: A=baseline, B=QLoRA only, C=QLoRA+RAG+VCD",
    )
    parser.add_argument("--data_root", default="./data", help="Root data directory")
    parser.add_argument(
        "--lora_path",
        default="checkpoints/lora_weights",
        help="Path to LoRA checkpoint directory",
    )
    parser.add_argument(
        "--faiss_index",
        default="data/faiss_index",
        help="Path to FAISS index directory",
    )
    parser.add_argument(
        "--output_file",
        default=None,
        help="Output JSON file (default: results/inference_{mode}.json)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of items (useful for smoke tests)",
    )
    parser.add_argument(
        "--dataset",
        choices=["vqav2", "pope"],
        default="vqav2",
        help="Dataset to run inference on",
    )
    parser.add_argument(
        "--pope_mode",
        choices=["random", "popular", "adversarial"],
        default="random",
        help="POPE subset mode (used when --dataset pope)",
    )
    parser.add_argument(
        "--config_file",
        default="config/inference_config.yaml",
        help="Path to inference_config.yaml",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint file for batch resume (JSON)",
    )

    return parser.parse_args()


def load_dataset(args: argparse.Namespace):
    """Load dataset items based on --dataset flag.

    Returns:
        Tuple of (images, questions, question_ids, image_paths)
    """
    from PIL import Image as PILImage

    if args.dataset == "vqav2":
        from src.data import VQAv2Loader
        logger.info(f"Loading VQAv2 (val split) from {args.data_root}...")
        loader = VQAv2Loader(data_root=args.data_root)
        data = loader.load(split="val")
        items = list(data.values())
        if args.limit:
            items = items[: args.limit]

        images, questions, question_ids, image_paths = [], [], [], []
        for item in items:
            img_path = item.get("image_path", "")
            try:
                img = PILImage.open(img_path).convert("RGB")
            except Exception:
                img = PILImage.new("RGB", (224, 224), color=(128, 128, 128))
            images.append(img)
            questions.append(item.get("question", ""))
            question_ids.append(item.get("question_id"))
            image_paths.append(img_path)

    elif args.dataset == "pope":
        from src.data import POPELoader
        logger.info(f"Loading POPE ({args.pope_mode}) from {args.data_root}...")
        loader = POPELoader(data_root=args.data_root)
        items = loader.load(mode=args.pope_mode)
        if args.limit:
            items = items[: args.limit]

        images, questions, question_ids, image_paths = [], [], [], []
        coco_val_dir = Path(args.data_root) / "vqav2" / "val"
        for i, item in enumerate(items):
            img_file = item.get("image", "")
            img_path = str(coco_val_dir / img_file)
            try:
                img = PILImage.open(img_path).convert("RGB")
            except Exception:
                img = PILImage.new("RGB", (224, 224), color=(64, 128, 64))
            images.append(img)
            questions.append(item.get("text", ""))
            question_ids.append(item.get("question_id", i))
            image_paths.append(img_path)

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    logger.info(f"Loaded {len(images)} items from {args.dataset}.")
    return images, questions, question_ids, image_paths


def main() -> None:
    """Entry point for inference CLI."""
    args = parse_args()

    output_file = args.output_file or f"results/inference_{args.mode}.json"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    checkpoint_file = args.checkpoint or f"results/batch_checkpoints/inference_{args.mode}_ckpt.json"

    # Build InferenceConfig
    config_path = Path(args.config_file)
    if config_path.exists():
        config = InferenceConfig.from_yaml(str(config_path))
        # CLI overrides
        config.lora_path = args.lora_path
        config.faiss_index_path = args.faiss_index
        config.seed = args.seed
    else:
        logger.warning(f"Config file not found: {config_path}. Using defaults.")
        config = InferenceConfig(
            lora_path=args.lora_path,
            faiss_index_path=args.faiss_index,
            seed=args.seed,
        )

    # Inject pre-loaded FAISSRetriever
    faiss_path = Path(args.faiss_index)
    if faiss_path.exists():
        config.faiss_retriever = FAISSRetriever(str(faiss_path))
        logger.info(f"FAISSRetriever loaded from {faiss_path}")
    else:
        logger.warning(f"FAISS index not found at {faiss_path}. RAG will be disabled.")

    # Load dataset
    images, questions, question_ids, image_paths = load_dataset(args)

    # Run batch inference
    pipeline = InferencePipeline(config)
    start = time.perf_counter()

    logger.info(f"Starting batch inference: mode={args.mode}, n={len(images)}")
    results = pipeline.run_batch(
        images=images,
        questions=questions,
        mode=args.mode,
        checkpoint_file=checkpoint_file,
        question_ids=question_ids,
        image_paths=image_paths,
    )
    total_time = time.perf_counter() - start

    # Compute summary stats
    valid = [r for r in results if r.answer not in ("[OOM]", "[ERROR]")]
    avg_latency = sum(r.latency_seconds for r in valid) / len(valid) if valid else 0.0
    avg_memory = sum(r.gpu_memory_gb for r in valid) / len(valid) if valid else 0.0
    rag_rate = sum(1 for r in valid if r.rag_triggered) / len(valid) if valid else 0.0

    summary = {
        "mode": args.mode,
        "dataset": args.dataset,
        "total_items": len(results),
        "valid_items": len(valid),
        "total_time_seconds": round(total_time, 2),
        "avg_latency_seconds": round(avg_latency, 3),
        "avg_memory_gb": round(avg_memory, 3),
        "rag_activation_rate": round(rag_rate, 4),
        "output_file": output_file,
    }

    # Save results
    output = {"summary": summary, "results": [r.to_dict() for r in results]}
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    # Print summary
    logger.info("=" * 60)
    logger.info(f"INFERENCE COMPLETE — Mode {args.mode}")
    logger.info(f"  Total time:    {total_time:.1f}s")
    logger.info(f"  Items:         {len(valid)}/{len(results)} valid")
    logger.info(f"  Avg latency:   {avg_latency:.3f}s/image")
    logger.info(f"  Avg memory:    {avg_memory:.2f} GB")
    logger.info(f"  RAG activation:{rag_rate:.1%}")
    logger.info(f"  Output:        {output_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
