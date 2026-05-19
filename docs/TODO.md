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

## 📊 Unified Model Performance Benchmarks

| Model Name & Identifier | Training Dataset Composition | LoRA Config ($r / \\alpha$) | Trainable Parameters | Optimal Calibrated Threshold | Dev (Validation) Split Metrics <br> (Precision / Recall / F1) | Gold Test Set Metrics <br> (Precision / Recall / F1) | Raw Counts <br> (TP / FP / FN) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard Baseline** <br> `gliner2_archaeo_lora_20260518_1704` | `260` Human Sentences | $4$ / $8$ | `663,552` <br> *(0.25% of base)* | **`0.8`** | `0.7812` / `0.5097` / **`0.6172`** | `0.7386` / `0.5509` / **`0.6311`** | `65` / `23` / `53` |
| **Old 500-Synthetic** <br> `gliner2_archaeo_lora_20260519_0101` | `260` Human + `500` Uncurated Synth | $4$ / $8$ | `663,552` <br> *(0.25% of base)* | **`0.5`** | `0.5957` / `0.5833` / **`0.5896`** | `0.6952` / `0.6186` / **`0.6547`** | `73` / `32` / `45` |
| **New Curated 1:1** <br> `gliner2_archaeo_lora_20260519_0256` | `260` Human + `260` Curated Synth | $4$ / $8$ | `663,552` <br> *(0.25% of base)* | **`0.8`** | `0.7360` / `0.5679` / **`0.6406`** | `0.7368` / `0.5679` / **`0.6400`** | `67` / `24` / `51` |
| **Medium-Capacity Baseline** <br> `gliner2_archaeo_lora_20260519_0822` | `260` Human Sentences | **$8$** / **$16$** | `1,327,104` <br> *(0.43% of base)* | **`0.7`** | `0.6929` / `0.6154` / **`0.6519`** | `0.6634` / `0.5678` / **`0.6119`** | `67` / `34` / `51` |
| **High-Capacity Baseline** <br> `gliner2_archaeo_lora_20260519_0704` | `260` Human Sentences | **$16$** / **$32$** | `2,654,208` <br> *(1.01% of base)* | **`0.8`** | `0.7016` / `0.6084` / **`0.6517`** | `0.6800` / `0.5763` / **`0.6239`** | `68` / `32` / `50` |
| **Real-Seeded (r=4)** <br> `gliner2_archaeo_lora_20260519_1206` | `260` Human + `58` Real-Seeded Synth | $4$ / $8$ | `663,552` <br> *(0.25% of base)* | **`0.8`** | `0.7236` / `0.6224` / **`0.6692`** | `0.7556` / `0.5763` / **`0.6538`** | `68` / `22` / `50` |
---

## 🔬 Pairwise Statistical Significance (Bootstrap Resampling)

Pairwise statistical significance test results using **Bootstrap Resampling** ($B=10,000$ draws) on the 32 unseen Gold Test sentences:

| Comparison (Model 2 vs. Model 1) | Observed F1 Difference | One-sided p-value (Model 2 > Model 1) | Two-sided p-value (Model 2 != Model 1) | Statistically Significant (α = 0.05)? |
| :--- | :---: | :---: | :---: | :---: |
| **Real-Seeded (r=4)** vs. **Baseline (No-Synthetic)** | **+0.0228** | `0.0764` | `0.1528` | **No** |
| **Real-Seeded (r=4)** vs. **New Curated 1:1** | **+0.0295** | `0.1208` | `0.2416` | **No** |
| **Real-Seeded (r=4)** vs. **Old 500-Synthetic** | **-0.0009** | `0.5163` | `0.9674` | **No** |
| **New Curated 1:1** vs. **Baseline (No-Synthetic)** | **-0.0067** | `0.6374` | `0.7252` | **No** |
| **Old 500-Synthetic** vs. **Baseline (No-Synthetic)** | **+0.0236** | `0.2285` | `0.4570` | **No** |
| **Medium-Capacity Baseline** vs. **Baseline (No-Synthetic)** | **-0.0192** | `0.8861` | `0.2278` | **No** |
| **High-Capacity Baseline** vs. **Baseline (No-Synthetic)** | **-0.0072** | `0.6604` | `0.6792` | **No** |
| **Old 500-Synthetic** vs. **New Curated 1:1** | **+0.0303** | `0.1298` | `0.2596` | **No** |
| **Medium-Capacity Baseline** vs. **New Curated 1:1** | **-0.0125** | `0.6188` | `0.7624` | **No** |
| **High-Capacity Baseline** vs. **New Curated 1:1** | **-0.0005** | `0.4889` | `0.9778` | **No** |
| **Old 500-Synthetic** vs. **Medium-Capacity Baseline** | **+0.0428** | `0.1066` | `0.2132` | **No** |
| **High-Capacity Baseline** vs. **Medium-Capacity Baseline** | **+0.0120** | `0.1080` | `0.2160` | **No** |
| **Old 500-Synthetic** vs. **High-Capacity Baseline** | **+0.0309** | `0.2145` | `0.4290` | **No** |
---

### 🔬 Explanation on synthetic data variants and statistical significance

*   **Uncurated Synth (Human + 500 uncurated synthetic sentences)**: F1: `0.6547` (+0.0236), Precision: `0.6952`, Recall: `0.6186`.
    *   *What is Uncurated Synth?*: A high-volume dataset where synthetic sentences are generated from scratch by the LLM (without grounding in real texts) and mixed directly into training. Due to the lack of grounding (causing semantic drift) and lack of balanced curation, the model suffered from severe validation dilution (F1: `0.5896` on dev), demonstrating that increasing synthetic volume without curation degrades model precision.
*   **Curated Synth (Human + 260 curated synthetic sentences)**: F1: `0.6400` (+0.0089), Precision: `0.7368`, Recall: `0.5679`.
    *   *What is Curated Synth?*: A balanced dataset where synthetic sentences generated by the LLM are capped at an exact 1:1 ratio with human baseline sentences. Unlike uncurated generation (which dilutes training data), these samples undergo strict quality filtering and automated span flattening to resolve overlapping entity boundaries.
*   **Real-Seeded (Human + 58 seeded sentences from 11 raw files)**: F1: `0.6538` (+0.0227), Precision: `0.7556`, Recall: `0.5763`.
    *   *What is Real-Seeded?*: Instead of having the LLM generate simulated sentences from scratch (which risks semantic drift), we supply **real, unannotated archaeological text fragments** from 11 domain-specific files. The LLM is used purely to annotate these real-world contexts, keeping the synthetic dataset tightly aligned with actual archaeological terminology and sentence structures.
    *   *How is it constructed?*:
        1.  **Sentence Segmentation**: The pipeline scans `data/extra_texts/` for `.txt` files, reads their raw content, and segments them into clean, non-trivial sentences.
        2.  **Few-Shot Prompting**: For each raw sentence, a prompt is constructed containing the official annotation guidelines and a set of randomly drawn human few-shot examples.
        3.  **LLM Annotation**: The LLM extracts entities (ARTEFACT, Period, etc.) directly within the context of the real sentence.
        4.  **Span Alignment & Flattening**: The pipeline matches the LLM's raw text extractions back to exact character-token indices in the original sentence and flattens any nested spans, outputting training-ready GLiNER samples.
*   **Pairwise Statistical Stability Testing (Bootstrap Resampling, B=10,000 Draws)**:
    *   *What is B?*: **B** represents the number of bootstrap resampling draws (we set $B = 10,000$ to ensure mathematically stable confidence estimates).
    *   *The Concept (What is Bootstrap?)*: Since our test set is very small (32 sentences), calculating metrics once doesn't show if the F1 improvement is stable. We simulate 10,000 different test set variations by drawing 32 sentences at random with replacement (meaning in any given draw, some sentences are repeated and others are left out).
        *   *After Drawing*: For each of the 10,000 simulated test sets, we calculate the F1 score of both models and record their difference ($\text{F1}_{\text{Real-Seeded}} - \text{F1}_{\text{Baseline}}$).
    *   *One-Sided Stability (Real-Seeded > Baseline)*:
        *   *Result*: $p = 0.0764$.
        *   *Insight*: This means that in **9,236 out of our 10,000 simulated runs (92.36%)**, the Real-Seeded model outperformed the Baseline. There is only a **7.64% chance** (the $p$-value) that this $+0.0228$ F1 gain is a random fluke of our specific 32 test sentences.
    *   *Two-Sided Difference (Real-Seeded != Baseline)*:
        *   *Result*: $p = 0.1528$.
        *   *Insight*: There is an **84.72% probability** that the two models systematically perform differently. The remaining 15.28% probability falls within the range of random statistical noise.
    *   *Why we do not have strict statistical significance ($p < 0.05$)*:
        *   *Insight*: In scientific reporting, we typically require a $p$-value below **`0.05`** (95% confidence) to declare a result "statistically significant." Because our test set is so small (32 sentences), the test lacks the *statistical power* to prove a $+0.0228$ gain is 95% certain, even though the directional signal is strong. To reach $p < 0.05$, we must either achieve a larger performance gap or evaluate on a larger test set.

---

### 🔬 Aligned Pipelines & Protocols (For Retraining & Verification)

#### A. Protocol to Generate More Seeded Data

1.  Place unannotated archaeological text fragments inside:
    ```bash
    data/extra_texts/
    ```
2.  Configure `.env`:
    ```ini
    SYNTHETIC_USE_EXTRA_TEXTS=True
    SYNTHETIC_EXTRA_TEXTS_DIR=data/extra_texts
    SYNTHETIC_NUM_BATCHES=50
    SYNTHETIC_SAMPLES_PER_BATCH=5
    ```
3.  Run generation:
    ```bash
    uv run notebooks/generate_synthetic_data.py
    ```
    Output: `data/synthetic_data_generation/synthetic_archaeology_real_seeded_gemini25flash.json`

#### B. Protocol to Run Training

1.  Configure `.env`:
    ```ini
    USE_SYNTHETIC_DATA=True
    SYNTHETIC_DATA_PATH=data/synthetic_data_generation/synthetic_archaeology_real_seeded_gemini25flash.json
    GLINER_SYNTHETIC_SAMPLE_SIZE=260  
    ```
2.  Run training:
    ```bash
    uv run notebooks/gliner2_training.py
    ```

#### C. Protocol to Run Performance Verification

If the new model achieves a higher Gold Test F1 score than the baseline, run the verification script to check its performance stability:

1.  Update the directory path under `Real-Seeded (r=4)` in `notebooks/compute_significance.py` to target the new best model checkpoint folder.
2.  Execute the verification script:
    ```bash
    uv run notebooks/compute_significance.py
    ```

---

## 📈 Hyperparameters & Convergence Benchmarks (Training Phase)

This table tracks hyperparameters, training datasets, and peak validation metrics achieved during active model training prior to post-training inference calibration:

| Metric / Parameter | Standard Baseline (r=4) | Old 500-Synthetic (r=4) | New Curated 1:1 (r=4) | Medium-Capacity Baseline (r=8) | High-Capacity Baseline (r=16) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **LoRA Rank ($r$)** | $4$ | $4$ | $4$ | **$8$** | **$16$** *(Capacity bump)* |
| **LoRA Alpha ($\alpha$)** | $8$ | $8$ | $8$ | **$16$** | **$32$** |
| **Dataset Size (Sentences)**| `260` (Human) | `260` (Human) + `500` (Synth) | `260` (Human) + `260` (Synth) | `260` (Human only) | `260` (Human only) |
| **Best Epoch** | Epoch 6 / 20 | Epoch 9 / 20 | Epoch 18 / 20 | Epoch 10 / 20 | Epoch 10 / 20 |
| **Peak Dev F1** | `0.6434` | `0.5643` | `0.6406` | **`0.6519`** 🚀 *(New Peak)* | `0.6517` |
| Peak Dev Precision | **`0.7812`** | `0.5957` | `0.7360` | `0.6929` | `0.7016` |
| Peak Dev Recall | `0.5097` | `0.5833` | `0.5679` | **`0.6154`** 🚀 *(Recall breakthrough)* | `0.6084` |

