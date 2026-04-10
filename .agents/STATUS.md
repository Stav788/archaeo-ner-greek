# Project Status: GLiNER2 Archaeological NER

## 1. Environment
* **Platform**: NVIDIA GB10 (Blackwell).
* **Stack**: `gliner2` v1.2.5, `torch` +cu130.
* **Model**: `fastino/gliner2-multi-v1`.
* **LoRA**: `r=4`, `alpha=8.0`, `dropout=0.1`.

## 2. Current Performance (2026-04-09)
* **Metric**: F1 ~**0.50** (P 0.40, R 0.67).
* **Configuration**: External JSON-based guidelines (`data/guidelines_*.json`).
* **Note**: English vs. Greek guidelines show marginal variance (F1 0.5083 vs 0.5000).



## 3. Task Board
- [ ] **Investigation**: Analyze low precision and mediocre recall causes.
