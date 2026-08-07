"""Standalone script to build CLIP embeddings and FAISS index for Module 1 RAG corpus.

Usage:
    python scripts/build_faiss_index.py --data_root ./data
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_data_root
from src.data import CLIPEmbeddingExtractor, CaptionLoader, FAISSIndexBuilder
from src.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Extract CLIP embeddings and build FAISS index.")
    parser.add_argument("--data_root", type=str, default=None, help="Root directory for dataset storage")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32", help="CLIP HF model")
    args = parser.parse_args()

    setup_logging("logs/build_faiss_index.log")

    if args.data_root:
        import os
        os.environ["DATA_ROOT"] = args.data_root

    data_root = get_data_root()
    index_dir = data_root / "faiss_index"

    print("==================================================")
    print("FAISS Vector Index Builder")
    print(f"Index Output Directory: {index_dir}")
    print("==================================================")

    # 1. Load Captions
    print("\n[1/3] Loading Captions Corpus...")
    caption_loader = CaptionLoader()
    captions = caption_loader.load()
    print(f"  ✓ Loaded {len(captions)} captions")

    # 2. Extract Embeddings
    print(f"\n[2/3] Extracting CLIP ({args.clip_model}) embeddings...")
    extractor = CLIPEmbeddingExtractor(model_name=args.clip_model)
    embeddings = extractor.extract(captions)
    print(f"  ✓ Extracted embeddings array of shape {embeddings.shape}")

    # 3. Build & Save FAISS Index
    print("\n[3/3] Building & Saving FAISS IndexFlatL2...")
    builder = FAISSIndexBuilder()
    index = builder.build(embeddings)
    builder.save(index, output_dir=index_dir, captions=captions)

    print("\n==================================================")
    print("FAISS Index Build Complete!")
    print("==================================================")


if __name__ == "__main__":
    main()
