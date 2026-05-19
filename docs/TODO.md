# To-do list διπλωματικής

| Task                             | Description                                                                                  |          Status          |
| :------------------------------- | :------------------------------------------------------------------------------------------- | :----------------------: |
| **SPECIES σε κάθε set**          | Αναπροσαρμογή split, ώστε η κατηγορία SPECIES να εκπροσωπείται σε train/validation/test set. |      ✅ Completed        |
| **Μεταφορά στο Hugging Face**    | Μεταφορά του έργου στην πλατφόρμα Hugging Face (`argilla` & `default` subsets).              |      ✅ Completed        |
| **Επανυπολογισμός του ΙΑΑ**      | Επανυπολογισμός του IAA με βάση το deduplicated dataset στο Hugging Face.                    |      ✅ Completed        |
| **Αναφορά πλήθους οντοτήτων**    | Εξαγωγή του αριθμού entities ανά label (ARTEFACT, PERIOD, LOCATION, κλπ) για τη διπλωματική. |  ✅ Completed (μέσω EDA) |
| **Υπολογισμός πλήθους κειμένων** | Καταγραφή του συνολικού αριθμού κειμένων και sentences στο corpus.                           |  ✅ Completed (μέσω EDA) |
| **Έλεγχος εκπροσώπησης**         | Υπολογισμός ποσοστού συμμετοχής κάθε label στο σύνολο των annotations.                       |  ✅ Completed (μέσω EDA) |
| **Επανεκπαίδευση**               | Επανεκπαίδευση του Gliner2-latest με LoRA (Βέλτιστο F1: 0.6311).                             |      ✅ Completed        |
| **Παραδείγματα κειμένων**        | Επιλογή αντιπροσωπευτικών αποσπασμάτων για παρουσίαση στη διπλωματική.                       |         ✅ Completed        |
| **Βελτιστοποίηση οδηγιών**       | Βελτιστοποίηση οδηγιών επισημείωσης, κυρίως για την κατηγορία PERIOD.                        |        ⏳ Pending        |
| **Καθάρισμα GitHub**             | Έλεγχος αρχείων, διαγραφή περιττών, οργάνωση φακέλων και καθαρό push στο dev branch.         |       |

---

## 🏃 1. Immediate Monitoring & Convergence (Next 1-2 Hours)
- [x] **Monitor the Curated 1:1 Training Run**:
  - Training is running in screen session: `screen -R archaeo-ner-greek-training`
  - Current state: **Epoch 20/20 Complete**.
  - Target: Let it run to complete Epoch 20.
- [x] **Verify End-of-Training Calibration Sweep**:
  - The script automatically triggered post-training threshold sweep on the best curated checkpoint.
  - Target: Capture the optimal calibrated threshold (`0.8`) and the corresponding F1/Recall/Precision scores on the **Gold Test Set**.
- [x] **Record Core Metrics**:
  - Extracted final TP, FP, FN counts from `tmp/training_20260519_0256.log` and `tmp/training_20260519_0704.log`.

---

## 🔬 2. Three-Way Statistical Significance ($p$-value) Suite
To publish or scientifically report these results, we must run pairwise significance tests using **Bootstrap Resampling** ($B=10,000$ draws) on the 32 test sentences:

- [ ] **Collect Raw Test Predictions**:
  - Run the three saved models on the Gold Test Set using their respective calibrated thresholds:
    1. **Baseline**: `threshold=0.8` (checkpoint in `data/models/gliner2_archaeo_lora_20260518_1704/best`)
    2. **Old 500-Synthetic**: `threshold=0.5` (checkpoint in `data/models/gliner2_archaeo_lora_20260519_0101/best`)
    3. **New Curated 1:1**: `threshold=TBD` (checkpoint in `data/models/gliner2_archaeo_lora_20260519_0256/best`)
  - Format the predictions at the sentence level: `{'tp': count, 'fp': count, 'fn': count}`.
- [ ] **Run Pairwise Significance Script**:
  - Run the bootstrap resampling test (with `B=10,000` draws) pairwise:
    - [ ] **Curated 1:1** vs. **Baseline** (Hypothesis: Clean synthetic data significantly improves over human-only).
    - [ ] **Curated 1:1** vs. **Old 500-Synthetic** (Hypothesis: Curation/balance significantly improves over raw volume).
    - [ ] **Old 500-Synthetic** vs. **Baseline** (Baseline reference check).
- [ ] **Build Significance Matrix**:
  - Document the absolute F1 gains and their corresponding $p$-values in a final benchmark table.

---

## 💾 3. Data Auditing & Final Reporting
- [ ] **Save Audit Results**:
  - Persist sentence-level prediction matches to `tmp/` or `data/` for qualitative error audits.
- [ ] **Update README.md**:
  - Document the final curated 1:1 F1 peak and the significance results under the project's Benchmark section.

---

## 📊 4. Unified Model Performance Benchmarks

| Model Name & Identifier | Training Dataset Composition | LoRA Config ($r / \alpha$) | Trainable Parameters | Optimal Calibrated Threshold | Dev (Validation) Split Metrics <br> (Precision / Recall / F1) | Gold Test Set Metrics <br> (Precision / Recall / F1) | Raw Counts <br> (TP / FP / FN) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard Baseline** <br> `gliner2_archaeo_lora_20260518_1704` | `260` Human Sentences | $4$ / $8$ | `663,552` <br> *(0.25% of base)* | **`0.8`** | `0.7812` / `0.5097` / **`0.6172`** | `0.7386` / `0.5509` / **`0.6311`** | `65` / `23` / `53` |
| **Old 500-Synthetic** <br> `gliner2_archaeo_lora_20260519_0101` | `260` Human + `500` Uncurated Synth | $4$ / $8$ | `663,552` <br> *(0.25% of base)* | **`0.5`** | `0.5957` / `0.5833` / **`0.5896`** | `0.6952` / `0.6186` / **`0.6547`** | `73` / `32` / `45` |
| **New Curated 1:1** <br> `gliner2_archaeo_lora_20260519_0256` | `260` Human + `260` Curated Synth | $4$ / $8$ | `663,552` <br> *(0.25% of base)* | **`0.8`** | `0.7360` / `0.5679` / **`0.6406`** | `0.7368` / `0.5679` / **`0.6400`** | `67` / `24` / `51` |
| **High-Capacity Baseline** <br> `gliner2_archaeo_lora_20260519_0704` | `260` Human Sentences | **$16$** / **$32$** | `2,654,208` <br> *(1.01% of base)* | **`0.8`** | `0.7016` / `0.6084` / **`0.6517`** | `0.6800` / `0.5763` / **`0.6239`** | `68` / `32` / `50` |

---

## 🔬 5. Pairwise Statistical Significance (Bootstrap Resampling)

Pairwise statistical significance test results using **Bootstrap Resampling** ($B=10,000$ draws) on the 32 unseen Gold Test sentences. Comparisons involving the High-Capacity Baseline are left blank/empty until significance evaluation is executed:

| Comparison (Model B vs. Model A) | Observed F1 Difference | One-sided p-value (B > A) | Two-sided p-value (B != A) | Statistically Significant (α = 0.05)? |
| :--- | :---: | :---: | :---: | :---: |
| **New Curated 1:1** vs. **Baseline (No-Synthetic)** | **-0.0067** | `0.6249` | `0.7502` | **No** |
| **Old 500-Synthetic** vs. **New Curated 1:1** | **+0.0303** | `0.1326` | `0.2652` | **No** |
| **Old 500-Synthetic** vs. **Baseline (No-Synthetic)** | **+0.0236** | `0.2326` | `0.4652` | **No** |
| **High-Capacity Baseline** vs. **Baseline (No-Synthetic)** | **-0.0072** | *[Pending]* | *[Pending]* | *[Pending]* |
| **High-Capacity Baseline** vs. **New Curated 1:1** | **-0.0161** | *[Pending]* | *[Pending]* | *[Pending]* |
| **High-Capacity Baseline** vs. **Old 500-Synthetic** | **-0.0308** | *[Pending]* | *[Pending]* | *[Pending]* |

---

## 📈 6. Hyperparameters & Convergence Benchmarks (Training Phase)

This table tracks hyperparameters, training datasets, and peak validation metrics achieved during active model training prior to post-training inference calibration:

| Metric / Parameter | Standard Baseline (r=4) | Old 500-Synthetic (r=4) | New Curated 1:1 (r=4) | High-Capacity Baseline (r=16) |
| :--- | :--- | :---: | :---: | :---: |
| **LoRA Rank ($r$)** | $4$ | $4$ | $4$ | **$16$** *(Capacity bump)* |
| **LoRA Alpha ($\alpha$)** | $8$ | $8$ | $8$ | **$32$** |
| **Dataset Size (Sentences)**| `260` (Human) | `260` (Human) + `500` (Synth) | `260` (Human) + `260` (Synth) | `260` (Human only) |
| **Best Epoch** | Epoch 6 / 20 | Epoch 9 / 20 | Epoch 18 / 20 | Epoch 10 / 20 |
| **Peak Dev F1** | `0.6434` | `0.5643` | `0.6406` | **`0.6517`** 🚀 *(New Peak)* |
| Peak Dev Precision | **`0.7812`** | `0.5957` | `0.7360` | `0.7016` |
| Peak Dev Recall | `0.5097` | `0.5833` | `0.5679` | **`0.6084`** 🚀 *(Recall breakthrough)* |

---

## 📋 7. Assistant Handover Status & Context

### 🏃‍♂️ Current Active Process (Background)
*   **Task**: Fine-tuning GLiNER2 with $r=8$ / $\alpha=16$ configuration.
*   **Screen Session**: `96324.archaeo-ner-greek-training`
*   **Active Log File**: [tmp/training_20260519_0822.log](file:///home/prokopis/src/archaeo-ner-greek/tmp/training_20260519_0822.log)
*   **W&B Live Dashboard**: [gliner2_archaeo_lora_20260519_0822 (rje0qby7)](https://wandb.ai/staalexandropoulou-national-and-kapodistrian-university-/archaeo-ner-greek/runs/rje0qby7)

---

### 💾 Step-by-Step Instructions for the Next Agent

1.  **Monitor & Capture $r=8$ Results**:
    *   Inspect `tmp/training_20260519_0822.log` or resume the screen session: `screen -r archaeo-ner-greek-training`.
    *   Wait for the 20 epochs to finish. The script will automatically execute a post-training threshold sweep on the validation split and print the optimal calibrated threshold, and the corresponding Precision, Recall, F1, and raw counts (TP, FP, FN) on the isolated Gold Test Set.
    *   Log these values into the **Part 4 (Benchmarks)** and **Part 6 (Hyperparameters)** tables in this document.

2.  **Run Paired Statistical Significance Testing**:
    *   Once the $r=8$ model checkpoint is saved under `data/models/gliner2_archaeo_lora_20260519_0822/best`, update `data/synthetic_data_generation/compute_significance.py` (which already contains $r=4$ and $r=16$ models) to include the path and threshold for the $r=8$ model.
    *   Execute the script from the workspace root on CPU to avoid CUDA conflicts:
        ```bash
        python3 -u data/synthetic_data_generation/compute_significance.py
        ```
    *   Fill in all pending cells in **Part 5 (Significance)** table of this document using the newly printed pairwise bootstrap $p$-values ($B=10,000$).

3.  **Perform Final Staging and Commit**:
    *   Check for any leftover temporary files.
    *   Stage and commit the completed `docs/TODO.md` and significance reports to the `dev` branch.
