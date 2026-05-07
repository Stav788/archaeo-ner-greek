# Implementation Plan: GLiNER2 Training Optimization

## Current Baselines (Experiment: `gliner2_archaeo_lora_20260410_1714`)
*   **Threshold**: 0.8
*   **Validation (Dev) Set**: F1 **0.7485** (P 0.8101, R 0.6957)
*   **Gold Test Set**: F1 **0.73 - 0.75**
*   **Note on Imbalance**: Previous test performance equaled or exceeded validation performance; likely attributed to imbalanced/random partitioning without document-grouping.

## Experiment: Balanced Grouped Split (Current)
*   **Best Seed (Best-of-100)**: **68**
*   **Split Distribution**: 
    *   **Train**: 274 samples (80.6%)
    *   **Val**: 35 samples (10.3%)
    *   **Test**: 31 samples (9.1%)
*   **Final Benchmark (Isolated Gold Test)**:
    *   **F1 Score**: **0.6960**
    *   **Precision**: 0.7900
    *   **Recall**: 0.6220
    *   **Final Training Loss (Epoch 30)**: **0.5691** (Total Loss)
    *   **Note**: This represents our true, leakage-protected baseline.

## Key Insights (Seed 68 Run)
*   **Zero Categorical Confusion**: The confusion matrix shows almost zero misclassifications between actual labels. The model understands the difference between `ARTEFACT`, `PERIOD`, `LOCATION`, etc., perfectly.
*   **The "Recall-Shy" Bottleneck**: Nearly 100% of the errors are **False Negatives** (misses). The model is extremely conservative, preferring to skip an entity rather than risk a wrong label.
*   **Primary Action**: Lower the global `THRESHOLD` (currently 0.8) to 0.6 or 0.5 in the next experiment to convert these "confident misses" into True Positives.

## 0. Modularization (Script Refactoring)
*   **Goal**: Reduce code duplication and improve readability.
*   **Actions**:
    *   - [x] **Environment Setup**: Modularize the Local/Colab initialization logic.
    *   - [x] **Data Preparation**: Create `df_to_gliner_examples()`.
    *   - [x] **Debugging**: Encapsulate verification logic.
    *   - [x] **Visualization**: Modularize plotting blocks.

## 1. Data Merging & Analysis
*   **Goal**: Create a unified pool and identify document boundaries.
*   **Actions**:
    *   - [x] **Unified Loading**: Merge `df_train` and `df_test` into `df_all` at the start of the script.
    *   - [x] **Grouping Analysis**: Identify `document_sentence_id_field` as the grouping key.
    *   - [x] **Extraction**: Implement `extract_doc_ids` utility to isolate the parent document ID.
    *   - [x] **Verification**: Verify data consistency across the merged pool.

## 2. Grouped Split Implementation
*   **Goal**: Implement 80/10/10 split without document leakage.
*   **Actions**:
    *   - [x] Create `grouped_split` utility function.
    *   - [x] Replace random shuffling logic with document-grouped 80/10/10 partitioning.
    *   - [x] Initialize separate `TrainingDataset` objects for Train, Val, and Test.
    *   - [x] Move `plot_ner_confusion_matrix` to `training_utils.py`.
    *   - [x] **Strict 80/10/10 Allocation**: Ensure final counts match target ratios.
    *   - [x] **Best-of-N Balanced Search**:
        *   Trial 100 random seeds to find the most balanced distribution.
        *   Evaluate balance by calculating the sentence-count deviation from the ideal 80/10/10 ratio.
        *   Select the seed with the minimum total deviation to ensure statistical representativeness.
        *   Persist the chosen seed in `split_manifest.json` for full reproducibility.

## 3. Training & Evaluation
*   **Goal**: Train on unified data and evaluate on the new isolated test split.
*   **Actions**:
    *   - [x] Update trainer to use `train_split` and `val_split`.
    *   - [x] Unify PRF reporting using the updated `get_cnt` utility.
    *   - [x] Implement final benchmark on isolated `test_split`.
    *   - [x] Modularize all remaining utility functions (`compute_metrics`, `evaluate_adapter`, `show_error_analysis`).
    *   - [x] Clean up all dead code and obsolete comments.

## 4. Synchronization
*   **Goal**: Maintain notebook compatibility.
*   **Actions**:
    *   - [x] Convert `gliner2_training.py` back to `.ipynb` using Jupytext.

## 5. Modular Utility Creation
*   **Goal**: Consolidate helper functions.
*   **Actions**:
    *   - [x] Move all `plot`, `verify`, and `split` functions to `training_utils.py`.

## 6. Experiment Tracking (WandB)
*   **Goal**: Implement real-time monitoring and experiment versioning.
*   **Actions**:
    *   - [x] Add `wandb` to environment dependencies.
    *   - [x] Initialize `wandb` run in `gliner2_training.py` using `experiment_name`.
    *   - [x] Implement a custom logging hook or callback for `GLiNER2Trainer` to sync metrics to WandB.
    *   - [x] Upload the best LoRA adapter as a WandB Artifact for reproducibility.

## Appendix: Metric & Loss Definitions
*   **Training Loss**: Calculated on the `train_split`. Used to update model weights.
*   **Evaluation Loss (`eval_loss`)**: Calculated on the `val_split`. Used for checkpoint selection and monitoring generalizability.
*   **Structure Loss (Span Loss)**: Measures boundary accuracy. Focuses on correctly identifying where an entity starts and ends.
*   **Entity Loss (Classification Loss)**: Measures label accuracy. Focuses on assigning the correct category (e.g., `LOCATION`) to a correctly identified span.
*   **Total Loss**: The weighted sum of Structure and Entity losses.

## Phase 2: Further Optimization & Synthetic Data
*   **Synthetic Data Generation**: 
    *   Develop augmentation scripts based strictly on the Training partition to avoid leakage.
    *   Implement entity swapping and LLM-driven re-contextualization to expand domain variety.
*   **Loss-Based Guidance**: 
    *   Experiment with `metric_for_best="eval_loss"` to compare generalizability against current F1-driven selection.
*   **Label Definition Refinement**: 
    *   Analyze confusion matrices to identify categorical overlaps and refine entity descriptions in JSON to resolve ambiguity.
