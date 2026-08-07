# Module 1 — Data Preparation & Infrastructure Setup

## Overview

Module 1 provides a clean, robust data infrastructure for loading **VQAv2**, **POPE**, and building a **FAISS vector retrieval index** over 1,000 MSCOCO training captions.

---

## Quickstart

```python
from src.data import VQAv2Loader, POPELoader, CaptionLoader, FAISSRetriever

# Load VQAv2 Dataset
vqa_loader = VQAv2Loader(data_root="./data")
train_dict = vqa_loader.load(split="train")

# Load POPE Evaluation Datasets
pope_loader = POPELoader(data_root="./data")
random_items = pope_loader.load(mode="random")
popular_items = pope_loader.load(mode="popular")
adversarial_items = pope_loader.load(mode="adversarial")

# Load RAG Caption Corpus
caption_loader = CaptionLoader()
captions = caption_loader.load()

# Query FAISS Index
retriever = FAISSRetriever(index_dir="./data/faiss_index")
top_3_captions = retriever.retrieve(query_embedding, k=3)
```

---

## Standalone Commands

```bash
# Download datasets & validate integrity
python scripts/download_datasets.py --data_root ./data --validate

# Build CLIP embeddings & FAISS index
python scripts/build_faiss_index.py --data_root ./data
```

---

## Architecture

```
src/data/
├── loaders.py        # VQAv2Loader, POPELoader, CaptionLoader
├── processors.py     # ImageProcessor, AnnotationProcessor, CLIPEmbeddingExtractor, DataValidator
└── retrieval.py      # FAISSIndexBuilder, FAISSRetriever
```

---

## Data Specifications

| Dataset | Size / Count | Details |
|---------|--------------|---------|
| VQAv2 Train | 82,783 images | MSCOCO train2014 images + VQA QA pairs |
| VQAv2 Val | 40,504 images | MSCOCO val2014 images + VQA QA pairs |
| POPE | 1,000 images / mode | Modes: `random`, `popular`, `adversarial` |
| Captions | 1,000 captions | 512-dim CLIP embeddings → `index.faiss` |
