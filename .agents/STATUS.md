# Project Status: GLiNER2 Archaeological NER

## 1. Environment
* **Platform**: NVIDIA GB10 (Blackwell).
* **Stack**: `gliner2` v1.2.5, `torch` +cu130.
* **Model**: `fastino/gliner2-multi-v1`.
* **LoRA**: `r=4`, `alpha=8.0`, `dropout=0.1`.

## 2. Current Performance (2026-04-10)
* **Metric**: F1 **0.7485** (P 0.8101, R 0.6957).
* **Experiment**: `gliner2_archaeo_lora_20260410_1714`.
* **Best State**: Threshold **0.8** (Optimized via sweep).
* **Dataset**: 212 samples (819 mentions).
* **Note**: Accuracy improved significantly by optimizing the inference threshold from 0.1 to 0.8.


## 3. Task Board
- [ ] **Data Expansion**: Annotate 21 additional samples as a **fixed "Gold" Test Set** (Held-out).
- [ ] **Split Optimization**: Transition to a 191/21/21 (Train/Val/Test) fixed-count split to prevent validation leak.
- [ ] **Final Benchmarking**: Report unbiased PRF metrics on the locked Gold Test Set.

## Qualitative Error Analysis Diagnostic

The model is identifying entities with high conceptual accuracy, but failing on technical "Exact Match" criteria in four specific areas:

### 1. The "Span Truncation" Problem (Primary cause of FNs)
The model often identifies the core entity but misses the descriptive adjectives or prepositions included in the ground truth.
*   **Example 3**: Missed "από το" in `από το 3600 ως το 3000 π.Χ.`.
*   **Example 12**: Missed "καμαροσκέπαστο" in `καμαροσκέπαστο θάλαμο`.
*   **Diagnostic**: This is a common NER challenge. The model is conceptually correct (it found the dating and the chamber), but technically "wrong" due to strict span matching.

### 2. Semantic Overlap: LOCATION vs. CONTEXT
There is significant confusion between ancient sites and specific buildings.
*   **Example 15**: Predicted `Πλωτινόπολης` as `CONTEXT` (building), but it was labeled as `LOCATION`.
*   **Example 3**: Predicted `όρμου του Αγίου Ιωάννη` as `CONTEXT` instead of `LOCATION`.
*   **Diagnostic**: In archaeology, sites are both locations and constructions. Large topographical features (bays, mountains, cities) should be strictly defined as `LOCATION`.

### 3. The "Dual Role" Entities
Certain entities change labels based on context, causing consistent misclassifications.
*   **Example 7**: `Spondylus gaederopus` (shell). Model predicted `SPECIES` (technically correct); annotator labeled it `MATERIAL` (functional role).
*   **Example 16**: `τέφρα` (ash). Model predicted `MATERIAL`; annotator wanted `ARTEFACT`.
*   **Diagnostic**: These are "valid but non-preferred" labels. The model is following biological traits over archaeological functions.

### 4. Semantic Anchoring Hallucinations (FPs)
*   **Example 5**: `Βασιλείου Πετράκου` (The archaeologist). The model predicted `CONTEXT`.
*   **Diagnostic**: Due to proximity to excavation site descriptions, the model is associating archaeologists' names with the `CONTEXT` category.

---

### Recommendations for Improvement

1.  **Refine Person Rule**: Explicitly state in instructions that "Archaeologists' names are always `PERSON`, never `CONTEXT`."
2.  **Clarify Geography**: Note that "Bays (όρμοι), peninsulas, and cities are `LOCATION` even if they contain remains."
3.  **Adjective Inclusion**: Update guidelines to specify that "Descriptive adjectives (καμαροσκέπαστος, κιβωτιόσχημος) must be included within the entity span."
