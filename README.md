# Hallucination-Reduced VQA

[![Tests](https://github.com/hasana157/hallucination-reduced-vqa/actions/workflows/tests.yml/badge.svg)](https://github.com/hasana157/hallucination-reduced-vqa/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Parameter-efficient hallucination reduction for Vision-Language Models.**
> Combines QLoRA fine-tuning, uncertainty-triggered RAG, and Visual Contrastive Decoding.
> Trainable on Google Colab T4 (free tier) in ~3 GPU hours.

---

## 🎯 Research Objective

Investigate the **connector bottleneck** in Vision-Language Models and demonstrate that
**QLoRA + RAG + VCD** can compensate for information loss introduced by the visual-language
connector module, reducing object hallucination by 42.9% (CHAIR: 0.210 → 0.120).

## 📊 Target Metrics

| Metric | Baseline | Target | Improvement |
|--------|----------|--------|-------------|
| VQA Accuracy | 65.2% | **71.5%** | +6.3% |
| CHAIR Score | 0.210 | **0.120** | -42.9% |
| POPE F1 | 0.52 | **0.68** | +30.8% |
| Inference Latency | — | 3.2s | — |
| Peak Memory | — | 14.1GB | Colab T4 ✓ |

## 🏗️ Architecture

```
VQAv2 + POPE + Captions
         │
    [Module 1] Data Preparation
         │
    FAISS Index ──────────────────┐
         │                        │
    [Module 2] QLoRA Training    [Module 3] Inference & RAG
         │                        │
    LoRA Weights ──────────────→ VCD + RAG Pipeline
                                  │
                             [Module 4] Evaluation
                                  │
                            VQA / CHAIR / POPE
```

## 🚀 Quickstart

### Prerequisites

- Python 3.9+
- CUDA 11.8+ (or Google Colab T4)
- ~20GB free storage for datasets

### Installation

```bash
git clone https://github.com/hasana157/hallucination-reduced-vqa.git
cd hallucination-reduced-vqa
pip install -r requirements.txt
```

### Module 1 — Data Preparation

```bash
# Download all datasets and build FAISS index
python scripts/download_datasets.py --data_root ./data --validate

# Build CLIP embeddings + FAISS index
python scripts/build_faiss_index.py --data_root ./data
```

### Module 2 — QLoRA Training

```bash
# Train Qwen2-VL-2B with QLoRA on VQAv2 (~3 GPU hours)
python scripts/train.py \
  --data_root ./data \
  --output_dir checkpoints/lora_weights \
  --num_epochs 2

# Resume from checkpoint if interrupted
python scripts/train.py \
  --data_root ./data \
  --resume_from_checkpoint checkpoints/lora_weights/checkpoint-5000
```

### Python API

```python
# Module 1 — Data loading
from src.data import VQAv2Loader, FAISSRetriever

loader = VQAv2Loader(data_root="./data")
train_data = loader.load(split="train")

retriever = FAISSRetriever(index_path="./data/faiss_index")
similar_captions = retriever.retrieve(query_embedding, k=3)

# Module 2 — Model loading
from src.models import QwenVLQuantizer, LoRAAdapter

quantizer = QwenVLQuantizer()
model = quantizer.load("Qwen/Qwen2-VL-2B-Instruct")
model = LoRAAdapter.apply(model)
```

## 📁 Repository Structure

```
hallucination-reduced-vqa/
├── config/                    # All configuration files
│   ├── default.yaml           # Master defaults
│   ├── data_config.yaml       # Dataset paths & CLIP settings
│   ├── training_config.yaml   # Training hyperparameters
│   └── lora_config.yaml       # LoRA-specific settings
├── src/
│   ├── config/                # Config loading utilities
│   ├── data/                  # Module 1: Data pipeline
│   │   ├── loaders.py         # VQAv2, POPE, Caption loaders
│   │   ├── processors.py      # CLIP embeddings, validation
│   │   └── retrieval.py       # FAISS index build & query
│   ├── models/                # Module 2: Model components
│   │   ├── qwen_vl.py         # 4-bit quantized Qwen2-VL
│   │   ├── lora_adapter.py    # PEFT LoRA adapter
│   │   └── vcd.py             # VCD (Module 3)
│   ├── training/              # Module 2: Training pipeline
│   │   ├── trainer.py         # QLoRA training loop
│   │   ├── callbacks.py       # Checkpointing, memory logging
│   │   └── utils.py           # Dataset, collator, metrics
│   ├── inference/             # Module 3: Inference pipeline
│   ├── evaluation/            # Module 4: Metrics & evaluation
│   └── utils/                 # Shared utilities
├── scripts/
│   ├── download_datasets.py   # Download VQAv2, POPE, captions
│   ├── build_faiss_index.py   # Build CLIP embeddings + FAISS
│   └── train.py               # QLoRA training entry point
├── notebooks/
│   ├── 01_setup_and_data.ipynb    # Colab: data setup
│   └── 02_qlora_training.ipynb    # Colab: training
├── tests/
│   ├── test_data.py           # Module 1 tests
│   └── test_training.py       # Module 2 tests
├── data/
│   └── captions/
│       └── training_captions.json  # 1000 RAG captions
├── checkpoints/               # LoRA weights (git-ignored)
├── results/                   # Evaluation outputs
└── docs/
    ├── DATA.md                # Data module documentation
    └── TRAINING.md            # Training module documentation
```

## 📓 Colab Notebooks

| Notebook | Description | Colab |
|----------|-------------|-------|
| `01_setup_and_data.ipynb` | Mount Drive, download datasets, build FAISS | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/01_setup_and_data.ipynb) |
| `02_qlora_training.ipynb` | QLoRA fine-tuning on T4 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/02_qlora_training.ipynb) |

## 🔬 Running Tests

```bash
# All tests
pytest tests/ -v

# Module 1 only
pytest tests/test_data.py -v --cov=src.data

# Module 2 only
pytest tests/test_training.py -v --cov=src.models,src.training

# Code quality
flake8 src/ scripts/
```

## ⚙️ Configuration

Set the `DATA_ROOT` environment variable before running:

```bash
# Local
export DATA_ROOT="./data"

# Google Colab
import os
os.environ["DATA_ROOT"] = "/content/gdrive/My Drive/hallucination-reduced-vqa/data"
```

All config is in `config/` — see [`config/data_config.yaml`](config/data_config.yaml)
and [`config/training_config.yaml`](config/training_config.yaml).

## 📜 Citation

If you use this code in your research, please cite:

```bibtex
@misc{hallucination-reduced-vqa,
  title   = {Hallucination-Reduced VQA: Parameter-Efficient Grounding with QLoRA, RAG and VCD},
  author  = {Hasan},
  year    = {2026},
  url     = {https://github.com/hasana157/hallucination-reduced-vqa}
}
```

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

> **Note**: VQAv2 and POPE datasets have their own licenses.
> This code is for academic/research use only.
