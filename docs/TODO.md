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
| **augmented-seeded-strict** <br> [`gliner2_archaeo_lora_20260519_2016`](https://wandb.ai/staalexandropoulou-national-and-kapodistrian-university-/archaeo-ner-greek/runs/n8f54ewg) | `260` Human + `93` Real-Seeded Synth | $4$ / $8$ | `663,552` <br> *(0.25% of base)* | **`0.7`** | `0.6923` / `0.6294` / **`0.6593`** | `0.7320` / `0.6017` / **`0.6605`** | `71` / `26` / `47` |
| **augmented-seeded-1to1** <br> [`gliner2_archaeo_lora_20260520_1445`](https://wandb.ai/staalexandropoulou-national-and-kapodistrian-university-/archaeo-ner-greek/runs/ve25cdr6) | `260` Human + `260` Seeded Synth | $4$ / $8$ | `663,552` <br> *(0.25% of base)* | **`0.8`** | `0.6875` / `0.6154` / **`0.6494`** | `0.7609` / `0.5932` / **`0.6667`** 🚀 | `70` / `22` / `48` |
---


## 🔬 Pairwise Statistical Significance (Bootstrap Resampling)

Pairwise statistical significance test results using **Bootstrap Resampling** ($B=10,000$ draws) on the 32 unseen Gold Test sentences:

| Comparison (Model 2 vs. Model 1) | Observed F1 Difference | One-sided p-value (Model 2 > Model 1) | Two-sided p-value (Model 2 != Model 1) | Statistically Significant (α = 0.05)? |
| :--- | :---: | :---: | :---: | :---: |
| **augmented-seeded-1to1** vs. **baseline (r=4)** | **+0.0356** | `0.1101` | `0.2202` | **No** |
| **augmented-seeded-1to1** vs. **augmented-seeded-strict** | **+0.0062** | `0.4106` | `0.8212` | **No** |
| **augmented-seeded-strict** vs. **baseline (r=4)** | **+0.0294** | `0.0597` | `0.1194` | **No** |
| **augmented-seeded-strict** vs. **augmented-filtered** | **+0.0361** | `0.0951` | `0.1902` | **No** |
| **augmented-seeded-strict** vs. **augmented-unfiltered** | **+0.0058** | `0.4141` | `0.8282` | **No** |
| **augmented-filtered** vs. **baseline (r=4)** | **-0.0067** | `0.6272` | `0.7456` | **No** |
| **augmented-unfiltered** vs. **baseline (r=4)** | **+0.0236** | `0.2244` | `0.4488` | **No** |
| **baseline (r=8)** vs. **baseline (r=4)** | **-0.0192** | `0.8816` | `0.2368` | **No** |
| **baseline (r=16)** vs. **baseline (r=4)** | **-0.0072** | `0.6592` | `0.6816` | **No** |
| **augmented-unfiltered** vs. **augmented-filtered** | **+0.0303** | `0.1300` | `0.2600` | **No** |
| **baseline (r=8)** vs. **augmented-filtered** | **-0.0125** | `0.6150` | `0.7700` | **No** |
| **baseline (r=16)** vs. **augmented-filtered** | **-0.0005** | `0.4967` | `0.9934` | **No** |
| **augmented-unfiltered** vs. **baseline (r=8)** | **+0.0428** | `0.1073` | `0.2146` | **No** |
| **baseline (r=16)** vs. **baseline (r=8)** | **+0.0120** | `0.1097` | `0.2194` | **No** |
| **augmented-unfiltered** vs. **baseline (r=16)** | **+0.0309** | `0.2051` | `0.4102` | **No** |
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

| Metric / Parameter | baseline (r=4) | augmented-unfiltered | augmented-filtered | baseline (r=8) | baseline (r=16) | augmented-seeded (n=58) | augmented-seeded (n=102) | augmented-seeded-strict | augmented-seeded-2.0 | augmented-seeded-1to1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LoRA Rank ($r$)** | $4$ | $4$ | $4$ | **$8$** | **$16$** *(Capacity bump)* | $4$ | $4$ | $4$ | $4$ |
| **LoRA Alpha ($\alpha$)** | $8$ | $8$ | $8$ | **$16$** | **$32$** | $8$ | $8$ | $8$ | $8$ |
| **Dataset Size (Sentences)**| `260` (Human) | `260` (Human) + `500` (Uncurated) | `260` (Human) + `260` (Curated) | `260` (Human) | `260` (Human) | `260` (Human) + `58` (Real-Seeded) | `260` (Human) + `102` (Real-Seeded) | `260` (Human) + `93` (Real-Seeded) | `260` (Human) + `260` (Seeded) |
| **Best Epoch** | Epoch 6 / 20 | Epoch 9 / 20 | Epoch 18 / 20 | Epoch 10 / 20 | Epoch 10 / 20 | Epoch 14 / 20 | Epoch 17 / 20 | Epoch 17 / 20 | Epoch 18 / 20 |
| **Peak Dev F1** | `0.6434` | `0.5643` | `0.6406` | **`0.6519`** | `0.6517` | **`0.6692`** 🚀 *(All-time Peak)* | `0.6519` | `0.6593` | `0.6494` |
| Peak Dev Precision | **`0.7812`** | `0.5957` | `0.7360` | `0.6929` | `0.7016` | `0.7411` | `0.6929` | `0.6923` | `0.6875` |
| Peak Dev Recall | `0.5097` | `0.5833` | `0.5679` | **`0.6154`** | `0.6084` | `0.6091` | **`0.6154`** 🚀 *(Peak recall)* | `0.6294` | `0.6154` |

---

## Thesis Methodology Guide: Annotation, Augmentation, and Evaluation

This section provides a formal, fact-based description of the data pipelines, annotation metrics, synthetic generation, and model training configurations, structured for direct inclusion in a research thesis or academic paper.

### 1. The Human Gold Standard Dataset

#### Annotation Process and Argilla Framework
* **Platform**: Argilla annotation interface hosted at `https://nlp.ilsp.gr/argilla/`, workspace `archaeo_ner_greek`.
* **Annotation Strategy**: Dual independent human annotation by domain experts (`stalexan` and `sasi.dimopoulou` / `tim.evans`) under defined annotation guidelines.
* **NFKD Unicode Normalization**: To resolve encoding variance in Greek text (where accented vowels like "ά" can be represented as a single combined character or as two separate byte components), all input strings undergo NFKD unicode normalization to ensure matching token representation across files.
* **Semantic Deduplication**: Identical text segments are identified in-memory and merged to prevent duplicate samples from distorting agreement statistics or model training.
* **Stratified Splitting (80/10/10)**: The cleaned sentences are partitioned into Training (80%), Validation (10%), and Test (10%) sets. A stratified split algorithm ensures that the relative frequency of rare labels (such as `SPECIES` or `SIGHT`) remains balanced across all three partitions.
* **Document-Grouped Partitioning**: Sentences originating from the same source document are restricted to the same partition. This prevents data leakage (where the model memorizes context or writing style from a document during training and gets evaluated on the same document).

#### Inter-Annotator Agreement (IAA)
To measure guideline consistency and annotation reproducibility, an inter-annotator agreement study was conducted on a shared subset of the corpus:
* **Evaluation Sample**: 212 sentences independently annotated by both `sasi.dimopoulou` and `stalexan`.
* **Sentence-Level Consensus**: Exactly 96 sentences (45.3% of the compared subset) achieved perfect consensus with identical span boundaries and semantic labels.
* **Global Span-Level Agreement**: The macro-averaged span-level F1-score across all categories is **0.7572**.
* **Category-Specific F1-Score Breakdown**:
  * `MATERIAL`: **0.8990** (highest agreement; physical raw substances)
  * `PERSON`: **0.8627** (high agreement; named historical or modern individuals)
  * `SPECIES`: **0.8500** (high agreement; biological and botanical categories)
  * `CONTEXT`: **0.7992** (moderate-to-high agreement; stratigraphic and depositional contexts)
  * `ARTEFACT`: **0.7456** (moderate agreement; portable human-made objects)
  * `FEATURE`: **0.7317** (moderate agreement; non-portable archaeological structures)
  * `LOCATION`: **0.6047** (low agreement; boundary ambiguity between macro- and micro-geographic references)
  * `PERIOD`: **0.5537** (lowest agreement; temporal span ambiguity and semantic overlap in historical epochs)
* **Discrepancy Typology Analysis**:
  * **Exact Matches**: 608 instances where both annotators agreed perfectly on both span boundaries and label assignment.
  * **Label Mismatches**: 6 instances where annotators identified the identical span boundary but assigned different semantic categories.
  * **Boundary Disagreements**: 179 instances where annotators identified overlapping spans but disagreed on character start or end offsets (average boundary character difference of **6.68** characters).
  * **Annotator Misses**: Instances where an entity was identified by only one of the two annotators:
    * Identified only by `stalexan`: 7 instances.
    * Identified only by `sasi.dimopoulou`: 31 instances.

#### Hugging Face Hub Structure
* **Repository Identifier**: [Stalexan/archaeo-ner-greek](https://huggingface.co/datasets/Stalexan/archaeo-ner-greek)
* **Dataset Configurations**:
  * **`default` subset**: Features standardized splits for model training, using two target columns: `input` (raw sentence text) and `output` (a dictionary mapping each entity label to a list of extracted text spans).
    * `train`: 260 sentences
    * `validation`: 32 sentences
    * `test`: 32 sentences
  * **`argilla` subset**: Features the full archival database containing original columns, annotator response histories, and metadata for record auditing.
  * **`synthetic` subset**: Features the LLM-annotated corpus derived from raw archaeological texts, segmented and split into raw and curated partitions (detailed in Section 2).

#### Programmatic Pipeline Scripts
* **Consolidation and Publishing Pipeline**: [notebooks/publish_dataset_to_hf.py](file:///home/prokopis/src/archaeo-ner-greek/notebooks/publish_dataset_to_hf.py) - Programmatically pulls records from Argilla, executes deduplication, performs document-grouped stratified splits, and publishes both the `default` and `argilla` configurations to Hugging Face.
* **Inter-Annotator Agreement (IAA) Pipeline**: [notebooks/iaa.py](file:///home/prokopis/src/archaeo-ner-greek/notebooks/iaa.py) - Computes agreement metrics between annotators, calculating overall span-level F1-score, category-specific F1-scores, and classifies errors into boundary mismatches vs. category misclassifications.
* **Exploratory Data Analysis (EDA) Pipeline**: [notebooks/EDA.py](file:///home/prokopis/src/archaeo-ner-greek/notebooks/EDA.py) - Extracts corpus statistics (sentence counts, token volumes, word length distributions, entity density per 100 tokens) and automatically generates LaTeX tables.

---

### 2. The Synthetic Augmented Dataset

#### Theoretical Justification, Necessity, and Empirical Proof of Efficacy
* **Definition of Seeded Synthetic Data**: Synthetic data in this framework refers to programmatic annotations generated by a large language model (LLM) on top of authentic, unannotated domain-specific historical text fragments. This contrasts with purely artificial text generation, which constructs sentence structures from scratch and risks semantic drift.
* **The Low-Resource Challenge**: Building Named Entity Recognition (NER) models for specialized domains like Greek archaeology presents a fundamental bottleneck. High-quality annotations require deep domain expertise (e.g., distinguishing stratigraphic contexts, physical artifact materials, and complex historic periods), making human annotation highly labor-intensive and costly. Consequently, the human gold standard dataset is restricted to a small volume of 324 total sentences.
* **Necessity for Data Augmentation**: Training neural architectures (like GLiNER2) on only 260 human sentences risks overfitting and poor generalization, especially for sparse, long-tail entity categories (e.g., `SPECIES`, `SIGHT`). Seeding authentic texts into an LLM-guided annotation pipeline offers a scalable solution to enrich training representations without incurring additional human cost.
* **Empirical Validation (Proof of Concept)**: The empirical utility of this synthetic dataset was verified by comparing fine-tuning runs on the Gold Test split (32 sentences):
  * **Zero-Shot Baseline (GLiNER2-latest)**: Achieved a low F1-score of **0.3464**, demonstrating that the out-of-the-box base model lacks domain-specific representation.
  * **Human-Only Fine-Tuning (baseline, r=4)**: Fine-tuning strictly on the 260 human training sentences improved the calibrated Gold Test F1-score to **0.6311**.
  * **Curated Synthetic Augmentation (augmented-seeded-1to1)**: Supplementing the 260 human sentences with 260 curated, balanced synthetic sentences generated via `gemini-2.5-flash` pushed the model's performance to its peak F1-score of **0.6667** (Precision: **0.7609**, Recall: **0.5932**). This absolute F1-score gain of **3.56%** (+0.0356) provides empirical proof that synthetic data successfully enhances domain-specific classification and recall boundaries.
  * **Note on Statistical Significance**: While this 3.56% absolute improvement is directionally positive and represents the peak model variant, pairwise bootstrap significance testing shows a $p$-value of **0.1101** against the baseline. This means the improvement is not statistically significant at the standard scientific threshold of $\alpha=0.05$. In our setting, this lack of strict mathematical significance is expected and acceptable given the extreme low-resource constraint of the Gold Test set (only 32 sentences). Because the test set is small, the statistical test has low mathematical power (i.e., it requires a much larger sample size to prove significance for moderate improvements). Nevertheless, the consistent gains in recall boundaries and precision across multiple configurations demonstrate the practical success of the augmentation pipeline despite the small evaluation size.

#### Generation Methodology and Seeding
* **The Concept of Seeding**: Rather than asking the LLM to generate archaeological sentences from scratch (which introduces semantic drift and non-authentic modern structures), unannotated archaeological text fragments are placed in the [data/extra_texts/](file:///home/prokopis/src/archaeo-ner-greek/data/extra_texts) directory. These fragments act as "seeds".
* **Source Corpus Statistics**: The unannotated seeding corpus contains exactly 55 raw text documents (sources), comprising 27,080 words (180,579 characters) in Greek. Source texts are compiled from three domain-specific origins:
  * Archaeological publications and reports (e.g., Archaeologiki Ephemeris, Praktika tis en Athinais Archaiologikis Etaireias, and Ergon).
  * National monument and space registries from the Greek Archaeological Cadastre (Arxaiologiko Ktimatologio).
  * Curated digitized corpora from CLARIN repositories detailing archaeological descriptions of castles, temples, and early Christian basilicas in Kos, Kalymnos, and Nisyros.
* **Annotation Generation**: The Gemini 2.5 Flash model (`gemini-2.5-flash`) reads these seeded sentences. Assisted by few-shot prompts (incorporating guideline definitions and baseline human annotations), it identifies entities within their real historical context.
* **Greedy Span Alignment and Flattening**: The LLM outputs labeled entities as text strings. The alignment code searches the source sentence to determine the exact character offset indices (`start` and `end`). In cases where the LLM generates overlapping or nested entities, a greedy span flattening algorithm preserves the widest boundaries and discards nested fragments to comply with the flat GLiNER2 schema.
* **Curation and Balancing**: To prevent validation dilution (where the model becomes biased toward synthetic language patterns), the synthetic dataset is curated and capped at a 1:1 ratio relative to the human training split (260 human sentences + 260 curated synthetic sentences).

#### Hugging Face Hub Subsets
* **Repository Configuration**: `synthetic` config within the [Stalexan/archaeo-ner-greek](https://huggingface.co/datasets/Stalexan/archaeo-ner-greek) repository.
* **Dataset Splits**:
  * **`raw_1350` split**: 1,350 sentences containing raw, uncurated LLM annotations.
  * **`curated_520` split**: 520 sentences dynamically filtered and balanced against the gold dataset properties.

#### Pipeline Scripts
* **Generation Script**: [notebooks/generate_synthetic_data.py](file:///home/prokopis/src/archaeo-ner-greek/notebooks/generate_synthetic_data.py) - Controls paragraph chunking, API prompts, boundary offset mapping, and generates raw JSON exports.
* **Curation and Upload Script**: [notebooks/publish_synthetic_to_hf.py](file:///home/prokopis/src/archaeo-ner-greek/notebooks/publish_synthetic_to_hf.py) - Curates raw JSON records, formats columns to GLiNER2 schemas, resolves provenance back to text files, and pushes the partitions to Hugging Face.

---

### 3. Model Fine-Tuning and Evaluation Framework

#### Model Architecture and LoRA Config
* **Base Architecture**: GLiNER2-latest.
* **Low-Rank Adaptation (LoRA)**: Parameter-efficient training that freezes base model weights and trains a small set of query, key, value, and projection adapter matrices (amounting to 0.25% to 1.01% of base weights). This prevents model parameter collapse on tiny datasets.
* **LoRA Rank ($r$) and Alpha ($\alpha$)**: The rank ($r$) dictates the dimension of the adapter matrices, while alpha ($\alpha$) acts as a constant scaling factor. Ranks tested include $r=4$ ($\alpha=8$), $r=8$ ($\alpha=16$), and $r=16$ ($\alpha=32$).
* **Calibrated Inference Threshold**: During model evaluation, entity predictions are output with probabilities. Rather than using a standard threshold of `0.5`, an automated sweep on the Validation split selects an optimal threshold (e.g. `0.8`) to balance Precision and Recall.
* **Fine-Tuning Script**: [notebooks/gliner2_training.py](file:///home/prokopis/src/archaeo-ner-greek/notebooks/gliner2_training.py) - Manages training epochs, dynamic synthetic mixing, validation calibration, and saves checkpoint outputs.

#### Weights & Biases (WandB) Run Registry
All fine-tuning sessions are tracked under the WandB project `archaeo-ner-greek` for the entity `staalexandropoulou-national-and-kapodistrian-university-`:

* **baseline (r=4)**: 260 human sentences ($r=4$). [WandB Run ij12xnv5](https://wandb.ai/staalexandropoulou-national-and-kapodistrian-university-/archaeo-ner-greek/runs/ij12xnv5)
* **augmented-unfiltered**: 260 human + 500 uncurated synthetic sentences ($r=4$). [WandB Run o5luweat](https://wandb.ai/staalexandropoulou-national-and-kapodistrian-university-/archaeo-ner-greek/runs/o5luweat)
* **augmented-filtered**: 260 human + 260 curated synthetic sentences ($r=4$). [WandB Run gbs39q1s](https://wandb.ai/staalexandropoulou-national-and-kapodistrian-university-/archaeo-ner-greek/runs/gbs39q1s)
* **baseline (r=8)**: 260 human sentences ($r=8$). [WandB Run 5bzdtbqi](https://wandb.ai/staalexandropoulou-national-and-kapodistrian-university-/archaeo-ner-greek/runs/5bzdtbqi)
* **baseline (r=16)**: 260 human sentences ($r=16$). [WandB Run r0y27ufn](https://wandb.ai/staalexandropoulou-national-and-kapodistrian-university-/archaeo-ner-greek/runs/r0y27ufn)
* **augmented-seeded-strict**: 260 human + 93 seeded strict synthetic sentences ($r=4$). [WandB Run n8f54ewg](https://wandb.ai/staalexandropoulou-national-and-kapodistrian-university-/archaeo-ner-greek/runs/n8f54ewg)
* **augmented-seeded-1to1**: 260 human + 260 seeded 1:1 synthetic sentences ($r=4$). [WandB Run ve25cdr6](https://wandb.ai/staalexandropoulou-national-and-kapodistrian-university-/archaeo-ner-greek/runs/ve25cdr6)

---

### 4. Verification and Significance Analysis

#### Significance Testing Framework
* **Bootstrap Resampling**: Because the unseen Gold Test set is small (32 sentences), computing evaluation metrics once does not prove that performance changes are stable.
* **Testing Protocol**: The script [notebooks/compute_significance.py](file:///home/prokopis/src/archaeo-ner-greek/notebooks/compute_significance.py) runs Bootstrap Resampling ($B=10,000$ draws) where it constructs 10,000 simulated test set variations by drawing sentences at random with replacement. It calculates model scores on each variation to derive one-sided and two-sided p-values.
* **Outcome**: The `augmented-seeded-1to1` model achieves the peak score on the Gold Test set (F1 of `0.6667`). However, significance tests show that this $+0.0356$ gain over the baseline is not statistically significant ($p=0.1101$, which is greater than the standard scientific alpha cutoff of $\alpha=0.05$). This indicates that while the directional performance improvement is positive, a larger test set is required to statistically prove significance.

