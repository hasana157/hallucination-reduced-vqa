#!/usr/bin/env python3
"""Evaluation CLI: run metrics computation and Connector Bottleneck Analysis.

Computes VQA accuracy, CHAIR score, POPE F1, latency, and memory across modes A, B, and C.
Generates comparison table markdown and connector bottleneck report JSON.

Usage:
    python scripts/evaluate.py --mode all --sample_size 50 --output_dir results/

    python scripts/evaluate.py --mode C --pope_mode random --sample_size 100
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import POPELoader, VQAv2Loader
from src.evaluation import (
    ComparisonReport,
    ConnectorBottleneckAnalyzer,
    Evaluator,
    ReproducibilityChecker,
)
from src.inference import InferenceConfig, InferencePipeline, InferenceResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("scripts.evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VQA Evaluation & Connector Bottleneck Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["A", "B", "C", "all"],
        default="all",
        help="Mode to evaluate: A, B, C, or all",
    )
    parser.add_argument(
        "--pope_mode",
        choices=["random", "popular", "adversarial", "all"],
        default="all",
        help="POPE evaluation mode",
    )
    parser.add_argument("--data_root", default="./data", help="Root data directory")
    parser.add_argument(
        "--sample_size",
        type=int,
        default=50,
        help="Number of samples for evaluation (default: 50 for quick run)",
    )
    parser.add_argument("--output_dir", default="results", help="Directory to save outputs")
    parser.add_argument(
        "--check_reproducibility",
        action="store_true",
        help="Run reproducibility check across 3 runs",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_evaluation_data(args: argparse.Namespace):
    """Load evaluation datasets from Module 1 loaders."""
    from PIL import Image as PILImage

    logger.info(f"Loading VQAv2 val dataset (sample_size={args.sample_size})...")
    vqa_loader = VQAv2Loader(data_root=args.data_root)
    vqa_data = vqa_loader.load(split="val")

    items = list(vqa_data.values())[: args.sample_size]

    images = []
    questions = []
    vqa_ground_truth = []
    chair_ground_truth = []

    coco_val_dir = Path(args.data_root) / "vqav2" / "val"

    for item in items:
        img_path = item.get("image_path", "")
        try:
            img = PILImage.open(img_path).convert("RGB")
        except Exception:
            img = PILImage.new("RGB", (224, 224), color=(128, 128, 128))
        images.append(img)
        questions.append(item.get("question", "What is in the image?"))
        vqa_ground_truth.append(item.get("answers", ["object"]))
        chair_ground_truth.append(item.get("answers", ["object"]))

    # Load POPE datasets
    pope_loader = POPELoader(data_root=args.data_root)
    pope_datasets: Dict[str, Tuple[List[str], List[str]]] = {}

    pope_modes = (
        ["random", "popular", "adversarial"]
        if args.pope_mode == "all"
        else [args.pope_mode]
    )

    for p_mode in pope_modes:
        logger.info(f"Loading POPE {p_mode} dataset...")
        p_items = pope_loader.load(mode=p_mode)[: args.sample_size]
        p_q = [item.get("text", "") for item in p_items]
        p_lbl = [item.get("label", "yes") for item in p_items]
        pope_datasets[p_mode] = (p_q, p_lbl)

    return images, questions, vqa_ground_truth, chair_ground_truth, pope_datasets


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    images, questions, vqa_gt, chair_gt, pope_datasets = load_evaluation_data(args)

    # 2. Check for pre-computed inference cache JSON files
    cached_by_mode: Dict[str, List[InferenceResult]] = {}
    modes_to_eval = ["A", "B", "C"] if args.mode == "all" else [args.mode]

    for m in modes_to_eval:
        cache_path = output_dir / f"inference_{m}.json"
        if cache_path.exists():
            logger.info(f"Found cached inference results at {cache_path}")
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            raw_results = data.get("results", [])
            cached_by_mode[m] = [
                InferenceResult(
                    answer=r["answer"],
                    latency_seconds=r["latency_seconds"],
                    gpu_memory_gb=r["gpu_memory_gb"],
                    rag_triggered=r["rag_triggered"],
                    evidence_used=r.get("evidence_used"),
                    entropy_score=r.get("entropy_score"),
                    mode=r["mode"],
                    question_id=r.get("question_id"),
                    image_path=r.get("image_path"),
                )
                for r in raw_results[: len(images)]
            ]

    # 3. Instantiate pipeline if any mode missing cache
    pipeline = None
    if len(cached_by_mode) < len(modes_to_eval):
        config = InferenceConfig(seed=args.seed)
        pipeline = InferencePipeline(config)

    evaluator = Evaluator(inference_pipeline=pipeline)

    # 4. Run evaluation
    if args.mode == "all":
        comparison_report = evaluator.evaluate_all(
            images=images,
            questions=questions,
            vqa_ground_truth=vqa_gt,
            chair_ground_truth=chair_gt,
            pope_datasets=pope_datasets,
            cached_results_by_mode=cached_by_mode,
        )

        # Print & save comparison table
        table_md = evaluator.generate_comparison_table(comparison_report)
        print("\n" + table_md)

        with open(output_dir / "comparison_table.md", "w", encoding="utf-8") as f:
            f.write(table_md)

        with open(output_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
            json.dump(comparison_report.to_dict(), f, indent=2)

        # Connector Bottleneck Analysis
        results_a = cached_by_mode.get("A", [])
        results_c = cached_by_mode.get("C", [])
        if results_a and results_c:
            analyzer = ConnectorBottleneckAnalyzer()
            bottleneck_report = analyzer.analyze(
                results_a=results_a,
                results_c=results_c,
                questions=questions,
                ground_truth=vqa_gt,
            )
            with open(output_dir / "connector_bottleneck_report.json", "w", encoding="utf-8") as f:
                json.dump(bottleneck_report, f, indent=2)
            logger.info(f"Saved connector bottleneck report to {output_dir / 'connector_bottleneck_report.json'}")

    else:
        # Single mode evaluation
        single_report = evaluator.evaluate_configuration(
            mode=args.mode,
            images=images,
            questions=questions,
            vqa_ground_truth=vqa_gt,
            chair_ground_truth=chair_gt,
            pope_datasets=pope_datasets,
            cached_results=cached_by_mode.get(args.mode),
        )
        out_file = output_dir / f"metrics_summary_{args.mode}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(single_report.to_dict(), f, indent=2)
        logger.info(f"Saved mode {args.mode} metrics to {out_file}")

    # 5. Reproducibility check if requested
    if args.check_reproducibility:
        logger.info("Running Reproducibility Check (SRS NFG-4)...")
        checker = ReproducibilityChecker(tolerance_pct=0.1, n_runs=3)

        def eval_wrapper():
            rep = evaluator.evaluate_configuration(
                mode="C" if "C" in modes_to_eval else modes_to_eval[0],
                images=images[:10],
                questions=questions[:10],
                vqa_ground_truth=vqa_gt[:10],
                chair_ground_truth=chair_gt[:10],
                pope_datasets={},
                cached_results=cached_by_mode.get("C", [])[:10] if "C" in cached_by_mode else None,
            )
            return {
                "vqa_accuracy": rep.vqa_accuracy,
                "chair_i": rep.chair_i,
                "latency": rep.avg_latency_seconds,
            }

        repro_res = checker.check(eval_wrapper)
        with open(output_dir / "reproducibility_report.json", "w", encoding="utf-8") as f:
            json.dump(repro_res, f, indent=2)
        logger.info(f"Reproducibility check passed: {repro_res['passed']}")


if __name__ == "__main__":
    main()
