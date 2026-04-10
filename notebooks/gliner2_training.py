# %% [markdown]
# # Create NER evaluation dataset

# %% [markdown]
# ## Init

# %%
# Standard library imports
import json
import logging
import warnings
from datetime import datetime
from logging.config import dictConfig
from pathlib import Path
from pprint import pprint

# Third-party imports
import argilla as rg
import pandas as pd
import seaborn as sns
import torch
from dotenv import dotenv_values, find_dotenv
from gliner2 import GLiNER2
from gliner2.training.data import InputExample, TrainingDataset
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

# Local project imports
from archaeo_ner_greek.logging_config import LOGGING_CONFIG
from archaeo_ner_greek.utils import (
    configure_argilla_client,
    get_dataset_as_dataframe,
)

# --- Environment & Configuration ---
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    from google.colab import userdata
    # Map Colab Secrets to env_vars
    env_vars = {
        "ARGILLA_API_URL": userdata.get("ARGILLA_API_URL"),
        "ARGILLA_API_KEY": userdata.get("ARGILLA_API_KEY"),
        "ARGILLA_WORKSPACE": userdata.get("ARGILLA_WORKSPACE"),
        "ARGILLA_DATASET": userdata.get("ARGILLA_DATASET"),
        "ANNOTATOR_A": userdata.get("ANNOTATOR_A"),
    }
    BASE_DIR = Path("/content/archaeo-ner-greek") # Default clone path
else:
    # Locate the root directory based on the .env file marker
    env_path = find_dotenv()
    if env_path:
        BASE_DIR = Path(env_path).resolve().parent
        env_vars = dotenv_values(env_path)
    else:
        BASE_DIR = Path.cwd().resolve()
        if BASE_DIR.name == "notebooks":
            BASE_DIR = BASE_DIR.parent
        env_vars = {}

# --- Hardware Check ---
assert torch.cuda.is_available(), "GPU still not detected"

# --- Logging Initialization ---
logging.getLogger("argilla").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="argilla")
dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# --- Project Setup ---
PROJECT_NAME = BASE_DIR.name
DATA_DIR = BASE_DIR / "data"
LOG_DIR = Path.home() / "logs"
sns.set_theme()

# Ensure required artifact directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f"Project root identified: {BASE_DIR}")
logger.info(f"Initialized project: {PROJECT_NAME}")
logger.info(f"Loaded environment variables: {list(env_vars.keys())}")


# %% [markdown]
# # 1. Data Loading and Preprocessing
# In this section, we load the raw archaeological documents (CSV/JSON) and prepare them for the NER pipeline. This includes text cleaning, context windowing, and splitting into train/test sets.
# 

# %%
def prepare_data(env_vars):
    DEFAULT_ANNOTATOR = env_vars.get("ANNOTATOR_A")
    df_annotated = get_dataset_as_dataframe(
        client=configure_argilla_client(env_vars=env_vars),
        dataset_name=env_vars.get("ARGILLA_DATASET"),
        workspace_name=env_vars.get("ARGILLA_WORKSPACE"), 
        username=DEFAULT_ANNOTATOR
    )
    if df_annotated.empty:
        raise ValueError("Annotated dataset is empty.")

    # Guidelines
    GUIDELINES_PATH = DATA_DIR / "guidelines_en.json" # Best performing profile
    with open(GUIDELINES_PATH, 'r', encoding='utf-8') as f:
        guidelines_dict = json.load(f)

    train_examples = []
    for _, row in df_annotated.iterrows():
        text = row['sentence_field']
        labels = row.get('labels', [])
        entities = {}
        for label_obj in labels:
            lbl = label_obj['label']
            start = label_obj['start']
            end = label_obj['end']
            mention = text[start:end].strip()
            if lbl not in entities: entities[lbl] = []
            entities[lbl].append(mention)
        
        train_examples.append(InputExample(text=text, entities=entities, entity_descriptions=guidelines_dict))
    
    train_dataset = TrainingDataset(train_examples)
    train_dataset.validate(raise_on_error=False)
    return train_dataset, guidelines_dict

# %%
print("Labelset and definitions:")
pprint(guidelines_dict)
pprint(train_dataset[0])

# %% [markdown]
# # 2. Model Training (GLiNER 2.0)
# We configure and execute the GLiNER 2.0 training process. This utilizes the semantic guidelines established for archaeological entities (ARTEFACT, PERIOD, etc.) to fine-tune the model on domain-specific Greek texts.
# 

# %%
experiment_name = f"gliner2_archaeo_lora_{datetime.now().strftime('%Y%m%d_%H%M')}"

# 1. Output Directory 
output_dir = DATA_DIR / "models" / experiment_name
output_dir.mkdir(parents=True, exist_ok=True)

# %%
# 2. LoRA Training Configuration
def compute_metrics(model, dataset, threshold=0.1):
    """Bulleted Micro-F1 calculation with local path-based schema loading."""
    import json # Ensure available in all scopes
    from pathlib import Path

    # RELOAD from disk to ensure consistency across process/worker boundaries
    # Using the path defined at top of script
    with open(GUIDELINES_PATH, 'r', encoding='utf-8') as f:
        current_guidelines = json.load(f)
    
    tp, fp, fn = 0, 0, 0
    model.eval()
    
    # Process examples
    examples = getattr(dataset, "examples", dataset)

    for i, ex in enumerate(examples):
        # 1. Retrieve text and schema
        if isinstance(ex, InputExample):
            text, gt_entities = ex.text, ex.entities
            # Prioritize ex.entity_descriptions if present, else fallback to current_guidelines
            schema = ex.entity_descriptions if (ex.entity_descriptions and len(ex.entity_descriptions) > 0) else current_guidelines
        elif isinstance(ex, (list, tuple)) and len(ex) >= 2:
            text, raw_schema = ex[0], ex[1]
            gt_entities = raw_schema.get("entities", {})
            desc = raw_schema.get("entity_descriptions", {})
            schema = desc if (desc and len(desc) > 0) else current_guidelines
        else:
            continue
            
        if not schema:
            raise RuntimeError(f"CRITICAL: Schema is EMPTY at index {i}. Guidelines failed to load or propagate.")

        if i == 0:
            print(f"\n--- EVAL CHECK (Step {getattr(model, 'global_step', 'N/A')}) ---")
            print(f"Schema Keys: {list(schema.keys())}")
            print(f"GT Entities Keys: {list(gt_entities.keys())}")
            import sys
            sys.stdout.flush()

        # 2. Extract and match
        output = model.extract_entities(text, schema, threshold=threshold)
        
        pred_spans = [(t, lbl) for lbl, texts in output.get('entities', {}).items() for t in texts]
        gt_spans = [(t, lbl) for lbl, texts in gt_entities.items() for t in texts]
                
        # 3. Exact Match Logic
        current_gt = gt_spans.copy()
        for p in pred_spans:
            if p in current_gt:
                tp += 1
                current_gt.remove(p)
            else:
                fp += 1
        fn += len(current_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {"f1": f1, "precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}
    print(f"\n>>> INTERIM EVAL: {metrics}")
    sys.stdout.flush()
    return metrics

def run_diagnostics(model, dataset, schema, threshold=0.1, output_prefix="diagnostic"):
    """
    Performs deep error analysis: Categorizes failures into boundary errors, 
    label confusions, and total misses. Generates a confusion matrix.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    import os

    all_labels = sorted(list(schema.keys()))
    label_to_idx = {lbl: i for i, lbl in enumerate(all_labels)}
    
    # We add a 'None/O' class for hallucinations and misses
    extended_labels = all_labels + ["O"]
    cm_data = [] # List of (gt_label, pred_label)

    error_registry = [] # Detailed logs of errors
    
    examples = getattr(dataset, "examples", dataset)
    print(f"Running diagnostics on {len(examples)} samples...")

    for idx, ex in enumerate(examples):
        text = ex.text
        # GT Format: {label: [mentions]} - need to resolve to offsets for precision
        # Since GLiNER2 InputExample usually stores mentions, we find them in text
        # WARNING: This assumes mentions are unique or we take first match (imperfect)
        # Note: In a real audit, we'd prefer character offsets from Argilla.
        gt_entities = ex.entities 
        
        # Predictive inference with spans
        # Assuming gliner2-multi-v1 supports return_spans=True
        try:
            # GLiNER2 .predict_entities is the offset-aware method
            # returns list of {'start': int, 'end': int, 'label': str, 'text': str, 'score': float}
            predictions = model.predict_entities(text, schema, threshold=threshold)
        except AttributeError:
            # Fallback to extract_entities if predict_entities is missing
            output = model.extract_entities(text, schema, threshold=threshold)
            # Reconstruct dummy spans from mentions (approximation)
            predictions = []
            for lbl, texts in output.get('entities', {}).items():
                for t in texts:
                    start = text.find(t)
                    if start != -1:
                        predictions.append({'start': start, 'end': start+len(t), 'label': lbl, 'text': t})

        # Heuristic to map GT mentions to spans
        gt_spans = []
        for lbl, texts in gt_entities.items():
            search_text = text
            offset = 0
            for t in texts:
                start = search_text.find(t)
                if start != -1:
                    gt_spans.append({'start': offset + start, 'end': offset + start + len(t), 'label': lbl, 'text': t, 'matched': False})
                    # Advance search index to handle multiple identical mentions
                    search_text = search_text[start + len(t):]
                    offset += start + len(t)

        # 1. Check Predictions against GT
        for p in predictions:
            matched_gt = None
            max_iou = 0
            for g in gt_spans:
                # Intersection
                inter_start = max(p['start'], g['start'])
                inter_end = min(p['end'], g['end'])
                if inter_end > inter_start:
                    union_start = min(p['start'], g['start'])
                    union_end = max(p['end'], g['end'])
                    iou = (inter_end - inter_start) / (union_end - union_start)
                    if iou > max_iou:
                        max_iou = iou
                        matched_gt = g
            
            if matched_gt:
                matched_gt['matched'] = True
                cm_data.append((matched_gt['label'], p['label']))
                
                error_type = "Correct" if (matched_gt['label'] == p['label'] and max_iou == 1.0) else \
                             "Boundary" if (matched_gt['label'] == p['label']) else \
                             "Confusion"
                
                if error_type != "Correct":
                    error_registry.append({
                        "type": error_type, "text": p['text'], "gt_label": matched_gt['label'], 
                        "pred_label": p['label'], "iou": max_iou, "context": text
                    })
            else:
                # Hallucination
                cm_data.append(("O", p['label']))
                error_registry.append({
                    "type": "Hallucination", "text": p['text'], "gt_label": "O", 
                    "pred_label": p['label'], "iou": 0, "context": text
                })

        # 2. Check for Misses
        for g in gt_spans:
            if not g['matched']:
                cm_data.append((g['label'], "O"))
                error_registry.append({
                    "type": "Miss", "text": g['text'], "gt_label": g['label'], 
                    "pred_label": "O", "iou": 0, "context": text
                })

    # Summary Generation
    err_df = pd.DataFrame(error_registry)
    summary = err_df['type'].value_counts()
    print("\n--- Error Type Summary ---")
    print(summary)
    
    # Save detailed CSV
    os.makedirs("tmp", exist_ok=True)
    err_df.to_csv(f"tmp/{output_prefix}_errors.csv", index=False)
    
    # Confusion Matrix Plot
    y_true, y_pred = zip(*cm_data) if cm_data else ([], [])
    if y_true:
        labels = sorted(list(set(y_true) | set(y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap="YlGnBu")
        plt.title(f"Confusion Matrix (Total Samples: {len(examples)})")
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(f"tmp/{output_prefix}_cm.png")
        print(f"Confusion matrix saved to tmp/{output_prefix}_cm.png")

    return err_df


# %%
# Training infrastructure moved to main block

# %% [markdown]
if __name__ == "__main__":
    import sys
    
    # 1. Prepare Data
    train_dataset, guidelines_dict = prepare_data(env_vars)
    train_split, val_split, _ = train_dataset.split(
        train_ratio=0.9, val_ratio=0.1, test_ratio=0.0, shuffle=True, seed=42
    )

    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    
    if mode == "train":
        experiment_name = f"gliner2_archaeo_lora_{datetime.now().strftime('%Y%m%d_%H%M')}"
        output_dir = DATA_DIR / "models" / experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)

        training_config = TrainingConfig(
            output_dir=str(output_dir), experiment_name=experiment_name, seed=42,
            batch_size=1, eval_batch_size=1, gradient_accumulation_steps=4, fp16=True,
            use_lora=True, lora_r=4, lora_alpha=8.0, lora_dropout=0.1,
            lora_target_modules=["encoder"], save_adapter_only=True,
            num_epochs=30, task_lr=1e-4, warmup_ratio=0.1, scheduler_type="cosine",
            weight_decay=0.01, eval_strategy="epoch", save_best=True,
            metric_for_best="f1", greater_is_better=True, save_total_limit=2, logging_steps=5
        )

        model = GLiNER2.from_pretrained("fastino/gliner2-multi-v1")
        trainer = GLiNER2Trainer(model, training_config, compute_metrics=compute_metrics)
        
        print("Starting training...")
        results = trainer.train(train_data=train_split, eval_data=val_split)
        
        best_path = Path(training_config.output_dir) / "best"
        model.load_adapter(str(best_path))
        run_diagnostics(model, val_split, guidelines_dict, output_prefix="eval_val")
        
    elif mode == "evaluate":
        specific_path = sys.argv[2] if len(sys.argv) > 2 else (DATA_DIR / "models/gliner2_archaeo_lora_20260409_1940/best")
        print(f"Diagnostic mode: Loading adapter from {specific_path}")
        
        eval_model = GLiNER2.from_pretrained("fastino/gliner2-multi-v1")
        eval_model.load_adapter(str(specific_path))
        
        run_diagnostics(eval_model, val_split, guidelines_dict, output_prefix="diagnostic_audit")

# %% [markdown]
# # 3. Evaluation and Performance Analysis
# Quantitative and qualitative assessment of the model's performance.
# - **Metrics**: Precision, Recall, and F1-score.
# - **Error Analysis**: Review of overlapping spans and MISC classification suggestions.
# 

# %%
def evaluate_adapter(model, adapter_path, test_data, threshold=0.1):
    """
    Loads a specific LoRA adapter and evaluates its performance.
    """
    # 1. Load the specific weights
    print(f"Loading adapter from: {adapter_path}")
    model.load_adapter(adapter_path)
    
    # 2. Execute metric calculation
    results = compute_metrics(model, test_data, threshold=threshold)
    
    print(f"\n--- EVALUATION RESULTS ({adapter_path.name}) threshold: {threshold} ---")
    print(f"F1 Score : {results['f1']}")
    print(f"Precision: {results['precision']}")
    print(f"Recall   : {results['recall']}")
    print(f"Counts   : TP={results['tp']}, FP={results['fp']}, FN={results['fn']}")
    
    return results
