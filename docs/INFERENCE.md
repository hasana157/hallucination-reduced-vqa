# Module 3: Inference & Grounding Pipeline (RAG + VCD)

## Overview

Module 3 implements the production-ready inference pipeline for **Hallucination-Reduced VQA**. It integrates three core components into a unified end-to-end architecture:

1. **Quantized Qwen2-VL-2B-Instruct Base Model + Trained LoRA Adapter** (from Module 2)
2. **Uncertainty-Triggered RAG**: Entropy-based confidence scoring that selectively retrieves evidence captions from the FAISS vector index (Module 1) only when the model is uncertain ($H_{\text{norm}} > 0.8$).
3. **Visual Contrastive Decoding (VCD)**: Training-free inference technique that contrasts logit distributions on the original image vs. a Gaussian-blurred image to eliminate language-prior co-occurrence hallucinations.

The system exposes a unified API (`InferencePipeline`) supporting the three configurations required by the **SRS Connector Bottleneck Analysis** (Section 2.3):

* **Mode A — Baseline**: Base model (no LoRA), greedy decoding, no RAG, no VCD.
* **Mode B — QLoRA Only**: LoRA adapter attached, greedy decoding, no RAG, no VCD.
* **Mode C — Proposed System**: LoRA adapter + Uncertainty-Triggered RAG + Visual Contrastive Decoding.

---

## Architecture Diagram

```
                              [Input Image + Question]
                                         │
                         ┌───────────────┴───────────────┐
                         │                               │
                      Mode A                           Mode B / C
             (Base Model, 4-bit)                  (QLoRA Model, 4-bit)
                         │                               │
                  Greedy Decode                   Entropy Scorer
                         │                      H(p) = -Σ p_i log(p_i)
                 [Answer Output]                         │
                                          ┌──────────────┴──────────────┐
                                          │                             │
                                  H <= 0.8 (Confident)          H > 0.8 (Uncertain)
                                          │                             │
                                    Skip Retrieval              Trigger RAG Retrieval
                                          │                     CLIP Embed -> FAISS k=3
                                          │                     "Evidence: cap1|cap2|cap3"
                                          │                             │
                                          └──────────────┬──────────────┘
                                                         │
                                           Visual Contrastive Decoder (VCD)
                                             Original Image -> Logits A
                                             Blurred Image  -> Logits B
                                             Logits = α·A - (1-α)·B
                                                         │
                                                Token-by-Token Loop
                                                         │
                                                  [Answer Output]
```

---

## Technical Specifications & Formulas

### 1. Entropy-Based Uncertainty Trigger

Given the first-token raw logits $z \in \mathbb{R}^{V}$ from a quick forward pass, we compute normalized Shannon entropy:

$$p_i = \text{softmax}(z_i) = \frac{\exp(z_i)}{\sum_{j=1}^{V} \exp(z_j)}$$

$$H(p) = -\sum_{i=1}^{V} p_i \log(p_i)$$

$$H_{\text{norm}} = \frac{H(p)}{\log(V)} \in [0, 1]$$

* **Trigger Rule**: If $H_{\text{norm}} > 0.8$, trigger retrieval.
* **Target Activation Rate**: $25\% - 30\%$ over a representative query batch (SRS FG-5).

### 2. Evidence Formatting

When retrieval fires, top-3 captions are retrieved via FAISS L2 similarity search on 512-dim CLIP text embeddings:

$$\text{Evidence Format: }\texttt{"Evidence: [cap1] | [cap2] | [cap3]"}$$

The evidence text is prepended to the prompt before generation.

### 3. Visual Contrastive Decoding (VCD)

For each token generation step $t$:

$$\text{Logits}_A = f_\theta(\mathbf{I}_{\text{orig}}, y_{<t}, q)$$

$$\text{Logits}_B = f_\theta(\mathbf{I}_{\text{blur}}, y_{<t}, q) \quad \text{where } \mathbf{I}_{\text{blur}} = \text{GaussianBlur}(\mathbf{I}_{\text{orig}}, r=15)$$

$$\text{Final Logits}_t = \alpha \cdot \text{Logits}_A - (1 - \alpha) \cdot \text{Logits}_B \quad (\alpha = 0.5)$$

$$y_t = \arg\max \left(\text{softmax}(\text{Final Logits}_t)\right)$$

* **Max VCD Overhead Budget**: $<500\text{ms}$ (SRS specification).

---

## Quickstart Usage

```python
from src.inference import InferencePipeline, InferenceConfig
from src.data import FAISSRetriever
from PIL import Image

# Initialize config
config = InferenceConfig(
    base_model="Qwen/Qwen2-VL-2B-Instruct",
    lora_path="checkpoints/lora_weights",
    faiss_index_path="data/faiss_index",
    entropy_threshold=0.8,
    vcd_alpha=0.5,
    blur_radius=15,
)

# Load pipeline
pipeline = InferencePipeline(config)

# Run single inference in Mode C (Proposed)
img = Image.open("sample.jpg").convert("RGB")
result = pipeline.run(img, "What is on the table?", mode="C")

print(f"Answer: {result.answer}")
print(f"Latency: {result.latency_seconds:.2f}s")
print(f"RAG Triggered: {result.rag_triggered}")
print(f"Evidence: {result.evidence_used}")
```

### CLI Batch Execution

```bash
# Run Mode C over VQAv2 split
python scripts/inference.py --mode C --data_root ./data --output_file results/inference_C.json --limit 100

# Run Mode A baseline
python scripts/inference.py --mode A --output_file results/inference_A.json --limit 100
```
