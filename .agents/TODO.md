# Project Status & Next Steps

### Status
- Baseline NER infrastructure.
- **Environment**: GPU-enabled `.venv` with **`gliner2`** (`+cu130`) successfully engaged on Blackwell (NVIDIA GB10).
- **Training**: Active 20-epoch LoRA run with custom metrics logic.

### Next Steps
- [x] **Baseline Capture**: Achieved F1 0.29 / Recall 0.60 @ threshold 0.1.
- [ ] **Guideline Refinement**: Remove mythology from `SPECIES`; tighten `PERSON` vs `ARTEFACT`.
- [ ] **Stability Run**: Execute `r=4`, `alpha=8.0`, `lr=1e-4` for 60 epochs.
- [ ] **Automated Logging**: Hook `compute_metrics` into `TrainingConfig` for epoch-by-epoch tracking.
- [ ] **State Debugging**: Identify and eliminate the race condition/cell sequence causing `guidelines_dict` to be periodically emptied.
- [ ] **Integration**: Prepare FastAPI model server handler.
