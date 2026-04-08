# Project Handover: GLiNER2 Archaeological NER

## 1. Current Progress
* **Hardware Recovery**: Resolved CPU-fallback by forcing `uv` to use `+cu130` wheels. NVIDIA GB10 (Grace-Blackwell) is now correctly engaged (46% utilization).
* **Environment**: `gliner2` v1.2.5 stable. `num_workers=0` enforced to avoid Python 3.12 multi-threading fork warnings.
* **Architecture**: Selected `fastino/gliner2-multi-v1` for superior Greek/multilingual support over `large-v1`.
* **LoRA Strategy**: 
    - Full-stack adaptation targeting `["encoder", "span_rep", "classifier", "count_embed", "count_pred"]`.
    - Configuration: `r=16`, `alpha=32`, `dropout=0.1`.
* **Metrics Implementation**: Developed custom `compute_metrics` hook for mention-based Micro-F1 calculation (missing in v1.2.5 core).
* **Execution Status**: In-progress 20-epoch run. `eval_loss` stagnated at Epoch 5.

## 2. Failure Diagnosis (2026-04-08)
* **Symptom**: `eval_loss` stops decreasing after 4-5 epochs on 187-sample dataset.
* **Root Cause A (Hyperparameters)**: `lora_r=16` is too high for <1K samples; creates excessive adaptation space.
* **Root Cause B (Learning Rate)**: `task_lr=5e-4` is too aggressive for small Greek niche dataset; likely overshooting.
* **Root Cause C (Validation)**: `raise_on_error=False` allows ingestion of potentially misaligned Greek spans.
* **Root Cause D (Metrics)**: `threshold=0.5` hides early "weak" signals from the model.

## 3. Recovery Strategy
* **Downscaling**: Reduce `lora_r` to 4 or 8. Use `lora_alpha = 2 * r`.
* **LR Reduction**: Lower `task_lr` to `1e-4` or `2e-4`.
* **Strict Validation**: Enforce `strict=True` to prune invalid spans before training starts.
* **Metric Sensitivity**: Lower evaluation threshold to `0.2` to track signal emergence.

## 4. Updated Configuration (Draft)
* **Split**: 90/10/0 (Train/Val/Test).
* **Stability**: Batch size 1, grad accumulation 4.

## 5. Baseline Performance (2026-04-08)
* **Run ID**: `gliner2_archaeo_lora_20260408_1654`
* **Metrics (Threshold 0.4)**:
    * **F1**: 0.3584
    * **Precision**: 0.2952
    * **Recall**: 0.4559
    * **Counts**: TP=31, FP=74, FN=37
* **Success**: Confirmed model captures `βωμός` (ARTEFACT) and `F. Cooper` (PERSON).

## 6. Major Fixes & Learnings
* **Schema Mismatch**: Fixed `compute_metrics` to handle dictionary output format `{'entities': {...}}`.
* **State Management**: Identified `guidelines_dict` variable confusion; consolidated source of truth.
* **Metric Function**: Implemented "Dual-Mode" `compute_metrics` for both training logs and notebook tests.
* **Synthetic Infrastructure**: Created `scripts/generate_synthetic_data.py` for OpenAI-powered data expansion.

## 7. Configuration for Tomorrow's Run
* **LoRA Rank**: `r=4`, `alpha=8.0` (Stricter adaptation for small dataset).
* **Learning Rate**: `1e-4` (Standard task adaptation).
* **Epochs**: `60` (Allow for more refinement).
* **Seed**: `42` (Fixed for cross-session consistency).

## 7. Key References
* Primary Notebook: `notebooks/gliner2_training.ipynb`
* Model Dir: `data/models/`
* Documentation: `docs/gliner2-tutorial/`
