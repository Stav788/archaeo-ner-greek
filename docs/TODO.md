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
| **Καθάρισμα GitHub**             | Έλεγχος αρχείων, διαγραφή περιττών, οργάνωση φακέλων και καθαρό push στο dev branch.         |        ⏳ Pending        |
| **Rerun Synthetic Push**          | Stavroula to rerun the push of the synthetic dataset (`notebooks/publish_synthetic_to_hf.py`).          |        ⏳ Pending        |
---

## 📊 Unified Model Performance Benchmarks

| Model Name & Identifier | Training Dataset Composition | LoRA Config ($r / \\alpha$) | Trainable Parameters | Optimal Calibrated Threshold | Dev (Validation) Split Metrics <br> (Precision / Recall / F1) | Gold Test Set Metrics <br> (Precision / Recall / F1) | Raw Counts <br> (TP / FP / FN) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **baseline (r=4)** <br> [`gliner2_archaeo_lora_20260518_1704`](https://wandb.ai/staalexandropoulou-national-and-kapodistrian-university-/archaeo-ner-greek/runs/ij12xnv5) | `260` Human Sentences | $4$ / $8$ | `663,552` <br> *(0.25% of base)* | **`0.8`** | `0.7812` / `0.5097` / **`0.6172`** | `0.7386` / `0.5509` / **`0.6311`** | `65` / `23` / `53` |
| **augmented-unfiltered** <br> [`gliner2_archaeo_lora_20260519_0101`](https://wandb.ai/staalexandropoulou-national-and-kapodistrian-university-/archaeo-ner-greek/runs/o5luweat) | `260` Human + `500` Uncurated Synth | $4$ / $8$ | `663,552` <br> *(0.25% of base)* | **`0.5`** | `0.5957` / `0.5833` / **`0.5896`** | `0.6952` / `0.6186` / **`0.6547`** | `73` / `32` / `45` |
| **augmented-filtered** <br> [`gliner2_archaeo_lora_20260519_0256`](https://wandb.ai/staalexandropoulou-national-and-kapodistrian-university-/archaeo-ner-greek/runs/gbs39q1s) | `260` Human + `260` Curated Synth | $4$ / $8$ | `663,552` <br> *(0.25% of base)* | **`0.8`** | `0.7360` / `0.5679` / **`0.6406`** | `0.7356` / `0.5424` / **`0.6244`** | `64` / `23` / `54` |
| **baseline (r=8)** <br> [`gliner2_archaeo_lora_20260519_0822`](https://wandb.ai/staalexandropoulou-national-and-kapodistrian-university-/archaeo-ner-greek/runs/5bzdtbqi) | `260` Human Sentences | **$8$ / $16$** | `1,327,104` <br> *(0.43% of base)* | **`0.7`** | `0.6929` / `0.6154` / **`0.6519`** | `0.6634` / `0.5678` / **`0.6119`** | `67` / `34` / `51` |
| **baseline (r=16)** <br> [`gliner2_archaeo_lora_20260519_0704`](https://wandb.ai/staalexandropoulou-national-and-kapodistrian-university-/archaeo-ner-greek/runs/r0y27ufn) | `260` Human Sentences | **$16$ / $32$** | `2,654,208` <br> *(1.01% of base)* | **`0.8`** | `0.7016` / `0.6084` / **`0.6517`** | `0.6800` / `0.5763` / **`0.6239`** | `68` / `32` / `50` |
| **augmented-seeded-strict** <br> [`gliner2_archaeo_lora_20260519_2016`](https://wandb.ai/staalexandropoulou-national-and-kapodistrian-university-/archaeo-ner-greek/runs/n8f54ewg) | `260` Human + `93` Real-Seeded Synth | $4$ / $8$ | `663,552` <br> *(0.25% of base)* | **`0.7`** | `0.6923` / `0.6294` / **`0.6593`** | `0.7320` / `0.6017` / **`0.6605`** 🚀 | `71` / `26` / `47` |
---


## 🔬 Pairwise Statistical Significance (Bootstrap Resampling)

Pairwise statistical significance test results using **Bootstrap Resampling** ($B=10,000$ draws) on the 32 unseen Gold Test sentences:

| Comparison (Model 2 vs. Model 1) | Observed F1 Difference | One-sided p-value (Model 2 > Model 1) | Two-sided p-value (Model 2 != Model 1) | Statistically Significant (α = 0.05)? |
| :--- | :---: | :---: | :---: | :---: |
| **augmented-seeded-strict** vs. **baseline (r=4)** | **+0.0294** | `0.0610` | `0.1220` | **No** |
| **augmented-seeded-strict** vs. **augmented-filtered** | **+0.0361** | `0.0912` | `0.1824` | **No** |
| **augmented-seeded-strict** vs. **augmented-unfiltered** | **+0.0058** | `0.4120` | `0.8240` | **No** |
| **augmented-filtered** vs. **baseline (r=4)** | **-0.0067** | `0.6374` | `0.7252` | **No** |
| **augmented-unfiltered** vs. **baseline (r=4)** | **+0.0236** | `0.2285` | `0.4570` | **No** |
| **baseline (r=8)** vs. **baseline (r=4)** | **-0.0192** | `0.8861` | `0.2278` | **No** |
| **baseline (r=16)** vs. **baseline (r=4)** | **-0.0072** | `0.6604` | `0.6792` | **No** |
| **augmented-unfiltered** vs. **augmented-filtered** | **+0.0303** | `0.1298` | `0.2596` | **No** |
| **baseline (r=8)** vs. **augmented-filtered** | **-0.0125** | `0.6188` | `0.7624` | **No** |
| **baseline (r=16)** vs. **augmented-filtered** | **-0.0005** | `0.4889` | `0.9778` | **No** |
| **augmented-unfiltered** vs. **baseline (r=8)** | **+0.0428** | `0.1066` | `0.2132` | **No** |
| **baseline (r=16)** vs. **baseline (r=8)** | **+0.0120** | `0.1080` | `0.2160` | **No** |
| **augmented-unfiltered** vs. **baseline (r=16)** | **+0.0309** | `0.2145` | `0.4290` | **No** |
### 🔬 Synthetic Data Variants & Statistical Significance
*   **augmented-unfiltered (Human + 500 uncurated synthetic sentences)**: F1: `0.6547` (+0.0236), Precision: `0.6952`, Recall: `0.6186`.
    *   *What is augmented-unfiltered?*: A dataset where synthetic sentences are generated by the LLM (without grounding in real texts) and mixed into training. Due to the lack of grounding (causing semantic drift) and lack of curation, the model suffered from validation dilution (F1: `0.5896` on dev), demonstrating that increasing synthetic volume without curation degrades model precision.
*   **augmented-filtered (Human + 260 curated synthetic sentences)**: F1: `0.6400` (+0.0089), Precision: `0.7368`, Recall: `0.5679`.
    *   *What is augmented-filtered?*: A dataset where synthetic sentences generated by the LLM are capped at a 1:1 ratio with human baseline sentences. These samples undergo quality filtering and automated span flattening to resolve overlapping entity boundaries.
*   **augmented-seeded (Human + seeded sentences from 11 raw files)**: F1: `0.6538` (+0.0227), Precision: `0.7556`, Recall: `0.5763`.
    *   *What is augmented-seeded?*: A dataset where text fragments from 11 files are annotated by the LLM. Keeping the synthetic dataset aligned with domain terminology and sentence structures. Only the variant (`augmented-seeded-strict`) enforces index alignment constraints during generation, yielding the peak F1 of `0.6605`.
    *   *How is it constructed?*:
        1.  **Sentence Segmentation**: The pipeline scans `data/extra_texts/` for `.txt` files, reads content, and segments them into sentences.
        2.  **Few-Shot Prompting**: For each sentence, a prompt is constructed containing the official annotation guidelines and human few-shot examples.
        3.  **LLM Annotation**: The LLM extracts entities (ARTEFACT, PERIOD, etc.) within the context of the sentence.
        4.  **Span Alignment & Flattening**: The pipeline matches the LLM's text extractions to character-token indices in the sentence and flattens nested spans, outputting training GLiNER samples.
*   **Pairwise Statistical Stability Testing (Bootstrap Resampling, B=10,000 Draws)**:
    *   *What is B?*: **B** represents the number of bootstrap resampling draws ($B = 10,000$).
    *   *The Concept (What is Bootstrap?)*: Since the test set is 32 sentences, calculating metrics once does not show if the F1 improvement is stable. We simulate 10,000 test set variations by drawing 32 sentences at random with replacement.
        *   *After Drawing*: For each of the 10,000 simulated test sets, we calculate the F1 score of both models and record their difference ($\text{F1}_{\text{augmented-seeded-strict}} - \text{F1}_{\text{baseline (r=4)}}$).
    *   *One-Sided Stability (augmented-seeded-strict > baseline (r=4))*:
        *   *Result*: $p = 0.0610$.
        *   *Insight*: This means that in **9,390 out of 10,000 simulated runs (93.90%)**, the `augmented-seeded-strict` model outperformed the `baseline (r=4)`. There is a **6.10% chance** (the $p$-value) that this $+0.0294$ F1 gain is due to the specific 32 test sentences.
    *   *Two-Sided Difference (augmented-seeded-strict != baseline (r=4))*:
        *   *Result*: $p = 0.1220$.
        *   *Insight*: There is an **87.80% probability** that the two models perform differently. The remaining 12.20% probability falls within statistical noise.
    *   *Why we do not have statistical significance ($p < 0.05$)*:
        *   *Insight*: In scientific reporting, we require a $p$-value below **`0.05`** to declare a result statistically significant. Because the test set is 32 sentences, the test lacks statistical power to prove a $+0.0294$ gain is certain. To reach $p < 0.05$, we must either achieve a larger performance gap or evaluate on a larger test set. the two models systematically perform differently. The remaining 12.20% probability falls within the range of random statistical noise.
    *   *Why we do not have strict statistical significance ($p < 0.05$)*:
        *   *Insight*: In scientific reporting, we typically require a $p$-value below **`0.05`** (95% confidence) to declare a result "statistically significant." Because our test set is so small (32 sentences), the test lacks the *statistical power* to prove a $+0.0294$ gain is 95% certain, even though the directional signal is strong. To reach $p < 0.05$, we must either achieve a larger performance gap or evaluate on a larger test set.

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
    SYNTHETIC_DATA_PATH=data/synthetic_data_generation/synthetic_archaeology_real_seeded_gemini25flash_n1350.json
    GLINER_SYNTHETIC_RATIO=2.0  # Dynamic synthetic-to-gold ratio (e.g. 2:1)
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

| Metric / Parameter | baseline (r=4) | augmented-unfiltered | augmented-filtered | baseline (r=8) | baseline (r=16) | augmented-seeded (n=58) | augmented-seeded (n=102) | augmented-seeded-strict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LoRA Rank ($r$)** | $4$ | $4$ | $4$ | **$8$** | **$16$** *(Capacity bump)* | $4$ | $4$ | $4$ |
| **LoRA Alpha ($\alpha$)** | $8$ | $8$ | $8$ | **$16$** | **$32$** | $8$ | $8$ | $8$ |
| **Dataset Size (Sentences)**| `260` (Human) | `260` (Human) + `500` (Uncurated) | `260` (Human) + `260` (Curated) | `260` (Human) | `260` (Human) | `260` (Human) + `58` (Real-Seeded) | `260` (Human) + `102` (Real-Seeded) | `260` (Human) + `93` (Real-Seeded) |
| **Best Epoch** | Epoch 6 / 20 | Epoch 9 / 20 | Epoch 18 / 20 | Epoch 10 / 20 | Epoch 10 / 20 | Epoch 14 / 20 | Epoch 17 / 20 | Epoch 17 / 20 |
| **Peak Dev F1** | `0.6434` | `0.5643` | `0.6406` | **`0.6519`** | `0.6517` | **`0.6692`** 🚀 *(All-time Peak)* | `0.6519` | `0.6593` |
| Peak Dev Precision | **`0.7812`** | `0.5957` | `0.7360` | `0.6929` | `0.7016` | `0.7411` | `0.6929` | `0.6923` |
| Peak Dev Recall | `0.5097` | `0.5833` | `0.5679` | **`0.6154`** | `0.6084` | `0.6091` | **`0.6154`** 🚀 *(Peak recall)* | `0.6294` |
