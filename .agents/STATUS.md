# Project Status: GLiNER2 Archaeological NER

## 1. Environment
* **Platform**: NVIDIA GB10 (Blackwell).
* **Stack**: `gliner2` v1.2.5, `torch` +cu130.
* **Model**: `fastino/gliner2-multi-v1`.
* **LoRA**: `r=4`, `alpha=8.0`, `dropout=0.1`.

## 2. Current Performance (2026-04-10)
* **Metric**: F1 **0.6829** (P 0.6195, R 0.7609).
* **Experiment**: `gliner2_archaeo_lora_20260410_1537`.
* **Best State**: Epoch 23/30.
* **Dataset**: 212 samples (819 mentions).
* **Note**: Switch to manual 90/10 split prevented sample loss (191 train / 21 val).

## 3. Task Board
- [ ] **Error Analysis**: Review 43 False Positives to identify semantic overlap.
- [ ] **Validation**: Test on separate unseen archaeological documents.

