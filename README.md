# Archaeo-NER-Greek

Archaeology today is confronted with a rapidly increasing volume of data, a large part of which takes the form of textual material. In the case of Greece, which has one of the densest archaeological landscapes worldwide, the volume of excavation reports, catalogues, and scholarly publications is particularly extensive. The systematic management and analysis of this information is becoming increasingly difficult using exclusively traditional methods. A promising approach for extracting structured information from archaeological texts is Named Entity Recognition (NER).

This work presents the development of a dataset specifically designed for Greek archaeology. A corpus consisting of 1,464 sentences was manually annotated according to a schema that includes eight entity types: Artefact, Context, Feature, Location, Material, Period, Person, and Species. To ensure quality, a subset of 324 sentences was independently annotated by two annotators, achieving an Inter-Annotator Agreement (IAA) $F_1$-score of 0.91 after the revision of the annotation guidelines, indicating high consistency in the annotation process.

The dataset was used to evaluate both fine-tuned small language models and state-of-the-art Large Language Models (LLMs) on the task of recognising archaeological entities in Greek texts. First, the GLiNER2 model (`fastino/gliner2-multi-v1`) was adapted via supervised fine-tuning, achieving a final micro-$F_1$ score of 0.77 on the human-annotated test set.

Additionally, the dataset served as a benchmark for evaluating modern instruction-tuned LLMs in zero-shot and few-shot settings. The results highlighted the challenges posed by specialised archaeological terminology, multi-word entities, linguistic ambiguity, and the syntactic complexity of Greek archaeological texts. In particular, the LLM evaluation demonstrated complex in-context learning dynamics, where smaller models benefited from few-shot demonstrations while massive models exhibited in-context demonstration bias.

Overall, this work aims to create the first specialised resource for the application of NER to Greek archaeology and to provide a basis for future research on the extraction and organisation of archaeological information from Greek texts.

## Dataset

The dataset is currently anonymized for peer review. The raw annotated data is provided in `data/archaeo_ner_greek.xlsx`. Upon acceptance, the dataset will be officially released via the Hugging Face Dataset Hub (the link is currently omitted to preserve double-blind anonymity). It consists of 1,464 manually annotated Greek archaeological sentences, which are distributed across the following splits:

- **Train**: 1,180 sentences
- **Validation**: 151 sentences
- **Test**: 133 sentences

## Evaluation

This repository provides scripts to evaluate the dataset using both the GLiNER2 fine-tuned approach and large language model (LLM) baselines. All scripts are executed in an isolated environment via `uv`.

### 1. GLiNER2 Evaluation

To train the model from scratch, or evaluate it on the test set, run the GLiNER2 training script.

To run a full training pipeline:

```bash
uv run python notebooks/gliner2_training.py
```

To evaluate an already fine-tuned adapter without retraining:

```bash
uv run python notebooks/gliner2_training.py --eval-only --adapter-path path/to/your/saved_adapter
```

### 2. LLM-based Evaluation

To evaluate the test dataset using various LLMs (0-shot and 5-shot), run the LLM evaluation script. The script relies on API providers (like OpenRouter) for large models and local inference for smaller domain models.

To run the evaluation across all models on the full dataset:

```bash
uv run python notebooks/llm_ner_evaluation.py --model all
```

You can also restrict the evaluation to a specific model or a subset of samples (useful for quick testing). For example:

```bash
# Example: Evaluate only 10 samples using a specific model (e.g., Llama 3)
uv run python notebooks/llm_ner_evaluation.py --model llama --samples 10
```

*Note: The evaluation script is pre-configured with the specific model endpoints. Passing short names like `llama`, `gemma`, `qwen`, or `krikri` automatically resolves to the exact model versions used in the paper (e.g., `meta-llama/llama-3.1-8b-instruct`).*
