# Notebooks

## `train_colab.ipynb` — Full Training & Evaluation Notebook

A complete, end-to-end Google Colab notebook for this project.

### What it does

| Step | Description |
|------|-------------|
| 0 | Verify GPU (T4 required) |
| 1 | Install all dependencies (`transformers`, `peft`, `bitsandbytes`, `faiss-cpu`, etc.) |
| 2 | Mount Google Drive (for persistent storage across sessions) |
| 3 | Clone / pull the repo from GitHub |
| 4 | Download VQAv2 via HuggingFace `datasets` + POPE from GitHub |
| 5 | Build FAISS index (CLIP-ViT-B-32 embeddings of 1000 captions) |
| 6 | QLoRA fine-tune Qwen2-VL-2B (2 epochs, ~3 GPU hours) |
| 7 | Auto-save checkpoints to Google Drive |
| 8 | Run inference — Modes A (baseline), B (QLoRA), C (QLoRA + RAG + VCD) |
| 9 | Evaluate: VQA accuracy, CHAIR, POPE F1, latency |
| 10 | Save all results to Drive + repo |
| 11 | Push results + adapter weights to GitHub |

---

### How to Use

1. **Open the notebook in Colab**  
   → Upload `train_colab.ipynb` to [colab.research.google.com](https://colab.research.google.com)  
   → Or click the badge below (if repo is public):  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hasana157/hallucination-reduced-vqa/blob/main/notebooks/train_colab.ipynb)

2. **Set runtime to GPU**  
   → `Runtime` → `Change runtime type` → `GPU` → `T4`

3. **Fill in your GitHub credentials** (Cell 3):
   ```python
   GITHUB_USERNAME = 'hasana157'       # your GitHub username
   GITHUB_REPO     = 'hallucination-reduced-vqa'
   GITHUB_TOKEN    = 'ghp_...'         # Personal Access Token (PAT)
   GITHUB_EMAIL    = 'your@email.com'
   ```

4. **Run all cells** (`Runtime` → `Run all`)

---

### Where Data is Saved

#### On Google Drive (`/content/drive/MyDrive/VQA_Project/`)
```
VQA_Project/
├── checkpoints/
│   └── lora_weights/
│       ├── adapter_model/          ← Final LoRA adapter (~50MB)
│       └── checkpoint-*/           ← Intermediate checkpoints
├── results/
│   ├── comparison_report.json      ← Full metrics for A/B/C
│   ├── comparison_table.md         ← Markdown table (copy into report)
│   ├── connector_bottleneck_analysis.json
│   └── srs_targets_summary.json    ← Pass/fail per SRS target
├── data/
│   └── faiss_index/               ← FAISS index files
└── logs/
    └── training/                  ← Training loss logs
```

#### On GitHub (after cell 11)
```
results/
├── comparison_report.json
├── comparison_table.md
├── connector_bottleneck_analysis.json
└── srs_targets_summary.json

checkpoints/lora_weights/adapter_model/
├── adapter_config.json
├── adapter_model.bin
└── README.md
```

---

### Resuming After Session Restart

Colab sessions time out after ~90 minutes of inactivity. To resume:

```python
# 1. Run setup cells (GPU, install, mount drive, clone repo)
# 2. Restore checkpoints from Drive:
import shutil
shutil.copytree(DRIVE_CHECKPOINTS, f'{REPO_DIR}/checkpoints', dirs_exist_ok=True)
shutil.copytree(f'{DRIVE_DATA}/faiss_index', f'{REPO_DIR}/data/faiss_index', dirs_exist_ok=True)
# 3. Continue from any cell (inference, evaluation, etc.)
```

---

### Getting a GitHub Personal Access Token (PAT)

1. Go to [github.com/settings/tokens](https://github.com/settings/tokens)
2. Click **Generate new token (classic)**
3. Give it a name (e.g., `colab-vqa`)
4. Select scopes: `repo` (full control of private/public repos)
5. Copy the token and paste it into the notebook as `GITHUB_TOKEN`

> ⚠️ Never commit your token to the repo. Paste it directly in the Colab cell only.

---

### Data Sources

| Dataset | Source | Size | Auto-downloaded? |
|---------|--------|------|-----------------|
| VQAv2 | HuggingFace `HuggingFaceM4/VQAv2` | ~25GB full / ~250MB subset | ✅ Yes (via `datasets` lib) |
| POPE (random) | GitHub raw JSON | ~500KB | ✅ Yes |
| POPE (popular) | GitHub raw JSON | ~500KB | ✅ Yes |
| POPE (adversarial) | GitHub raw JSON | ~500KB | ✅ Yes |
| CLIP-ViT-B-32 | HuggingFace `openai/clip-vit-base-patch32` | ~600MB | ✅ Yes (auto) |
| Qwen2-VL-2B | HuggingFace `Qwen/Qwen2-VL-2B-Instruct` | ~5.4GB (4-bit = ~1.4GB) | ✅ Yes (auto) |

**No manual downloads required** — everything is fetched automatically when you run the notebook.
