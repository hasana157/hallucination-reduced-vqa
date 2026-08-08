# Module 4: Evaluation & Results Analysis

## Overview

Module 4 provides the full evaluation suite and hypothesis testing framework for the **Hallucination-Reduced VQA** research pipeline. It evaluates the three system configurations (A: Baseline, B: QLoRA Only, C: Proposed System) across task accuracy, object hallucination rates, grounding quality, and computational efficiency.

---

## Core Metrics Definitions

### 1. VQA Accuracy (Task Performance)

Evaluates answer accuracy using the official VQAv2 scoring metric:

$$\text{Accuracy}(\text{ans}) = \min\left(\frac{\text{count}_{\text{human}}(\text{ans})}{3}, 1.0\right) \times 100\%$$

Answers are normalized (lowercased, stripped of punctuation and articles, number words converted to digits) before matching.

### 2. CHAIR Score (Object Hallucination Rate)

Measures object hallucinations by matching generated object mentions against ground-truth MSCOCO annotations using the 80 COCO classes and synonym mapping.

* **CHAIR_i (Instance-level)**: Fraction of hallucinated object mentions over all mentioned objects.

$$\text{CHAIR}_i = \frac{\sum_{j} |\text{Hallucinated Objects in Sentence } j|}{\sum_{j} |\text{Total Objects Mentioned in Sentence } j|}$$

* **CHAIR_s (Sentence-level)**: Fraction of answers containing at least one hallucinated object.

$$\text{CHAIR}_s = \frac{\sum_{j} \mathbb{I}(\text{Sentence } j \text{ contains } \ge 1 \text{ hallucinated object})}{\text{Total Sentences}}$$

### 3. POPE F1 Score (Grounding Quality)

Evaluates object existence grounding over Yes/No questions on MSCOCO across three splits:
1. **Random**: Random absent objects.
2. **Popular**: Frequently co-occurring absent objects.
3. **Adversarial**: Hardest co-occurring absent objects.

$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}$$

$$\text{POPE F1} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

## Target Metrics Summary Table

| Metric | Baseline (A) Target | Proposed System (C) Target | SRS Requirement |
| :--- | :---: | :---: | :---: |
| **VQA Accuracy** | ~68.2% | **≥ 70.0%** | SRS Spec |
| **CHAIR_i** | 0.210 | **≤ 0.168** | ≥20% Reduction |
| **POPE F1 (Avg)** | 0.520 | **≥ 0.600** | SRS Spec |
| **Inference Latency** | ~2.1s | **≤ 3.2s** | NFG-1 |
| **Peak GPU Memory** | ~12.5GB | **≤ 14.1GB** | NFG-2 |
| **Reproducibility** | N/A | **≤ ±0.1% Var** | NFG-4 |

---

## Connector Bottleneck Analysis (SRS Section 2.3)

Answers the two central research questions:

* **RQ-1 (Connector Contribution)**: Spatial pooling in vision-language connectors compresses visual features, creating an information bottleneck. When visual information is lost, the model defaults to language priors, hallucinating expected co-occurring objects (e.g. predicting "fork" when seeing "plate").
* **RQ-2 (Mitigation via RAG + VCD)**:
  * **RAG** provides external descriptive captions when first-token entropy is high ($H_{\text{norm}} > 0.8$).
  * **VCD** contrasts normal logits with blurred-image logits, subtracting the language-prior bias $\text{Logits}_B$ from $\text{Logits}_A$.

---

## CLI Execution

```bash
# Run full evaluation across all modes (A, B, C)
python scripts/evaluate.py --mode all --sample_size 100 --output_dir results/

# Run reproducibility check
python scripts/evaluate.py --mode all --check_reproducibility --output_dir results/
```
