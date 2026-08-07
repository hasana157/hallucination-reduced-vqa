"""Standalone script to download and validate VQAv2, POPE, and Captions datasets.

Usage:
    python scripts/download_datasets.py --data_root ./data --validate
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config, get_data_root
from src.data import CaptionLoader, DataValidator, POPELoader, VQAv2Loader
from src.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Download and validate VQAv2, POPE, and Captions datasets.")
    parser.add_argument("--data_root", type=str, default=None, help="Root directory for dataset storage")
    parser.add_argument("--validate", action="store_true", help="Run DataValidator checks after download")
    args = parser.parse_args()

    setup_logging("logs/download_datasets.log")

    if args.data_root:
        import os
        os.environ["DATA_ROOT"] = args.data_root

    data_root = get_data_root()
    print(f"==================================================")
    print(f"Data Preparation & Infrastructure Setup")
    print(f"Data Root: {data_root}")
    print(f"==================================================")

    # 1. Setup Captions
    print("\n[1/3] Loading 1000 Training Captions...")
    caption_loader = CaptionLoader()
    captions = caption_loader.load()
    print(f"  ✓ Captions loaded: {len(captions)} items")

    # 2. Setup POPE
    print("\n[2/3] Downloading POPE Datasets (random, popular, adversarial)...")
    pope_loader = POPELoader(data_root=data_root)
    for mode in ["random", "popular", "adversarial"]:
        pope_items = pope_loader.load(mode=mode)
        print(f"  ✓ POPE ({mode}): {len(pope_items)} items loaded")

    # 3. Setup VQAv2
    print("\n[3/3] Setting up VQAv2 Dataset (train/val splits)...")
    vqa_loader = VQAv2Loader(data_root=data_root)
    print("  Note: Full VQAv2 images/annotations download automatically on demand via HF datasets.")

    # 4. Validation
    if args.validate:
        print("\n[4/4] Running DataValidator checks...")
        validator = DataValidator(config={"data": {"root": str(data_root)}})
        report = validator.validate()
        print(f"  ✓ Data Validation Passed: {report['passed']}")

    print("\n==================================================")
    print("Module 1 Data Setup Complete!")
    print("==================================================")


if __name__ == "__main__":
    main()
