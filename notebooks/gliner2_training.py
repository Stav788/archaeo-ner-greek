# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Gliner2 training on archaeological NER data
#

# %% [markdown]
# ### Colab Configuration Requirements
# To run this notebook on Google Colab, add the following secrets to your environment (Key icon in the left sidebar):
# #### 1. Repository Access
# * **`GITHUB_TOKEN`**: Required for cloning private source code.
# * **Obtain**: [GitHub Settings](https://github.com/settings/tokens) > Developer Settings > Personal access tokens. Required scope: `repo` or `contents:read`.
# #### 2. Argilla Integration
# Retrievable from your Argilla instance profile page:
# * **`ARGILLA_API_URL`**: Instance endpoint.
# * **`ARGILLA_API_KEY`**: Personal API key.
# * **`ARGILLA_WORKSPACE`**: Target workspace.
# * **`ARGILLA_DATASET`**: Dataset name.
# * **`ANNOTATOR_A`**: Username associated with your annotations.
# #### 3. Activation
# * Toggle **Notebook access** to **ON** for all listed secrets.

# %% [markdown]
# ## Init

# %%
import os
import sys
import subprocess
from pathlib import Path
from training_utils import setup_local, setup_colab, df_to_gliner_examples, verify_annotations, plot_training_history, plot_threshold_curves, extract_doc_ids, grouped_split

# 1. Environment Detection & Pre-Import Setup
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    env_vars = setup_colab()
else:
    env_vars = setup_local()

# 2. Optimized Imports (Now safe because packages are installed/pathed)
import json
import logging
import warnings
import random
from datetime import datetime
from logging.config import dictConfig

from tabulate import tabulate
import matplotlib.pyplot as plt

import torch
from gliner2 import GLiNER2
from gliner2.training.data import InputExample, TrainingDataset
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Local project imports
import archaeo_ner_greek
from archaeo_ner_greek.logging_config import LOGGING_CONFIG
from archaeo_ner_greek.utils import (
    configure_argilla_client,
    get_dataset_as_dataframe,
)

# 3. Path Management
BASE_DIR = Path(os.getcwd())
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# 4. Logging & Global Config
dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings(action="ignore", message=r"datetime.datetime.utcnow")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*SwigPyObject.*")

logger.info(f">>> Working Directory: {BASE_DIR}")
logger.info(f">>> Models Directory:  {MODELS_DIR}")
logger.info(f">>> Dataset:  {env_vars['ARGILLA_DATASET']}")
logger.info(f">>> Test dataset: {env_vars['ARGILLA_TEST_DATASET']}")




# %% [markdown]
# ## Data Loading and Preprocessing
#

# %%
DEFAULT_ANNOTATOR = env_vars.get("ANNOTATOR_A")
client = configure_argilla_client(env_vars=env_vars)
workspace = env_vars.get("ARGILLA_WORKSPACE")

df_train = get_dataset_as_dataframe(
    client=client,
    dataset_name=env_vars.get("ARGILLA_DATASET"),
    workspace_name=workspace, 
    username=DEFAULT_ANNOTATOR
)

df_test = get_dataset_as_dataframe(
    client=client,
    dataset_name=env_vars.get("ARGILLA_TEST_DATASET"),
    workspace_name=workspace, 
    username=DEFAULT_ANNOTATOR
)

df_all = pd.concat([df_train, df_test], ignore_index=True)
logger.info(f"Merged Data: {len(df_all)} samples (Train: {len(df_train)}, Test: {len(df_test)})")

# Analysis for Grouped Splitting
logger.info("\n>>> DATASET ANALYSIS FOR GROUPING")
logger.info(f"Columns: {df_all.columns.tolist()}")
logger.info("Sample IDs (First 10):")
logger.info(df_all['id'].head(10).tolist())
logger.info("\nMetadata structure (if available):")
if 'metadata' in df_all.columns:
    logger.info(json.dumps(df_all['metadata'].iloc[0], indent=2))
else:
    logger.info("No 'metadata' column found.")

# Extract Parent Document IDs
df_all['doc_id'] = extract_doc_ids(df_all)
logger.info(f"Unique documents identified: {df_all['doc_id'].nunique()}")
logger.info("\n>>> DOC_ID MAPPING SAMPLE")
logger.info(df_all[['document_sentence_id_field', 'doc_id']].head(10))

# df_annotated = get_dataset_as_dataframe(
#     client=configure_argilla_client(env_vars=env_vars),
#     dataset_name=env_vars.get("ARGILLA_DATASET"),
#     workspace_name=env_vars.get("ARGILLA_WORKSPACE"), 
#     username=DEFAULT_ANNOTATOR
# )

if not df_all.empty:
    logger.info(f"Ready: {len(df_all)} samples loaded with 'labels' ready for training.")
    logger.info(f"Available Columns: {df_all.columns.tolist()}")

if not df_all.empty:
    row = df_all.iloc[0]
    logger.info(f"{'='*40} FULL ROW DEBUG {'='*40}")
    logger.info(f"ID     : {row['id']}")
    logger.info(f"Full Response Dict: {json.dumps(row['sentence_field'], indent=2, ensure_ascii=False)}")
    logger.info(f"Labels (Extracted): {row['labels']}")
    logger.debug(f"Full Response Dict: {json.dumps(row['response'], indent=2, ensure_ascii=False)}")

# %% [markdown]
# ### Guidelines to entities descriptions 

# %%
# Dynamically find the package resources folder
PACKAGE_ROOT = Path(archaeo_ner_greek.__file__).parent
RESOURCES_DIR = PACKAGE_ROOT / "resources"
GUIDELINES_PATH = RESOURCES_DIR / "archaeoner_labels_definitions_v7_st.json"
logger.info(f"Loading entity descriptions from {GUIDELINES_PATH}")
with open(GUIDELINES_PATH, 'r', encoding='utf-8') as f:
    entity_descriptions = json.load(f)

logger.info(f"Labels: {list(entity_descriptions.keys())}")
logger.info(f"Example: ARTEFACT: {entity_descriptions['ARTEFACT']}")

# %%
df_all.head(5)
# # Updated Verification block
target_ids = [
    "4d82e2e6-02c9-4fef-81f0-191ae553cb0f",
    "80811c3d-c530-4420-a660-da15a3459cfe",
    "2003_culture.gov_excavation_1371_10",
    "2025_nationalarchive_K7F58Y_598308_1"
]
targets = ["αστρικό κόσμημα", "τριπτά εργαλεία"]
pair_text = "παραστάδες"
pair_label = "FEATURE"

verify_annotations(df_all, target_ids, targets, pair_text, pair_label, DEFAULT_ANNOTATOR)

# %% [markdown]
# ### Document-level Grouped Split (80/10/10)

# %%
# Create grouped splits to ensure no document leakage
df_train, df_val, df_test = grouped_split(df_all, group_col='doc_id')
logger.info(f"Grouped Split Results: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")
# %% [markdown]
# ### Training examples

# %%
train_examples = df_to_gliner_examples(df_all, entity_descriptions)

logger.info(f"Text: {train_examples[20].text}")
logger.info(f"Entities: {train_examples[20].entities}")
logger.info(train_examples[20].entities)
logger.info(f'Entity descriptions example: ARTEFACT: {train_examples[20].entity_descriptions["ARTEFACT"]}')
logger.info(train_examples[20])




# %%
train_dataset = TrainingDataset(train_examples)
train_dataset.validate(raise_on_error=True)

# The following may throw away one example (Gliner2 bug)
# train_split, val_split, _ = train_dataset.split( 
#     train_ratio=0.9, 
#     val_ratio=0.1, 
#     test_ratio=0.0, 
#     shuffle=True, 
#     seed=42
# )

all_examples = train_dataset.examples.copy()
random.seed(42)
random.shuffle(all_examples)
val_size = int(len(all_examples) * 0.1) 
val_split = all_examples[:val_size]      
train_split = all_examples[val_size:]   
logger.info(f"Train: {len(train_split)} | Val: {len(val_split)} | Total: {len(train_split) + len(val_split)}")
train_split = TrainingDataset(train_split)
val_split = TrainingDataset(val_split)


for ds_name, ds in {"full": train_dataset, "train": train_split, "val":val_split}.items():
    logger.info(f"Dataset: {ds_name} ")
    ds.print_stats()
    logger.debug(ds[0])

# %% [markdown]
# # Training

# %% [markdown]
# ## Custom metrics for evaluation

# %%
THRESHOLD = 0.8

def compute_metrics(model, dataset, threshold=THRESHOLD):
    """Bulleted Micro-F1 calculation with local path-based schema loading."""
    tp, fp, fn = 0, 0, 0
    model.eval()

    for i, ex in enumerate(dataset):
        logger.debug(ex)
        # Inference
        text = ex[0]
        gt_entities = ex[1]["entities"] # Ground truth entities
        entity_descriptions = ex[1]["entity_descriptions"]

        output = model.extract_entities(text, entity_descriptions, threshold=threshold)
        pred_entities = output.get('entities', {})
        
        # Flatten pred spans
        pred_spans = []
        for lbl, texts in pred_entities.items():
            for t in texts:
                pred_spans.append((t, lbl))
        
        # Flatten gt spans
        gt_spans = []
        for lbl, texts in gt_entities.items():
            for t in texts:
                gt_spans.append((t, lbl))
        
        # Exact Match logic (order insensitive)
        temp_gt = gt_spans.copy()
        for p in pred_spans:
            if p in temp_gt:
                tp += 1
                temp_gt.remove(p)
            else:
                fp += 1
        fn += len(temp_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {"f1": f1, "precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}
    formatted_metrics = {k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in metrics.items()}
    logger.info(f"\n>>> EVAL: {formatted_metrics}")
    sys.stdout.flush()
    
    return metrics



# %% [markdown]
# ## Training config

# %%
experiment_name = f"gliner2_archaeo_lora_{datetime.now().strftime('%Y%m%d_%H%M')}"
output_dir = DATA_DIR / "models" / experiment_name
output_dir.mkdir(parents=True, exist_ok=True)
num_epochs = 30


training_config = TrainingConfig(
    output_dir=str(output_dir),
    experiment_name=experiment_name,
    seed=42,
    
    # Hardware & Batching Stability 
    batch_size=1,
    eval_batch_size=1,             # Prevents "tensor size mismatch" during evaluation
    gradient_accumulation_steps=4, # Simulates Effective Batch Size = 4
    fp16=True,                     # Half-precision for speed/memory
    
    # LoRA Architecture (Rank 4 for stability on small datasets)
    use_lora=True,
    lora_r=4,                     # Reduced from 16
    lora_alpha=8.0,               # Reduced from 32.0 (standard 2*r)
    lora_dropout=0.1,             # Regularization for small datasets
    lora_target_modules=["encoder"], # Focused target
    save_adapter_only=True,        # Saves ~10-30MB instead of 1.2GB per checkpoint

    # Optimization Profile
    num_epochs=num_epochs,
    task_lr=1e-4,                 # Primary learning rate for adapters/heads
    warmup_ratio=0.1,
    scheduler_type="cosine",       # Smooth decay for stable convergence
    weight_decay=0.01,


    # Checkpointing (Accuracy follows F1)
    eval_strategy="epoch",
    save_best=True,
    metric_for_best="f1",        # Use F1 to drive selection
    greater_is_better=True,      # Higher is better
    # metric_for_best="eval_loss", # Use Loss to drive selection
    # greater_is_better=False,      # Lower is better
    save_total_limit=2,
    logging_steps=5,
       
    # Early Stopping (DISABLED due to gliner2 v1.2.5 bug)
    early_stopping=False,
    early_stopping_patience=10,

    # Data Handling
    validate_data=True,
)

# %% [markdown]
# ## Trainer

# %%
model = GLiNER2.from_pretrained("fastino/gliner2-multi-v1") # Multi-tasking, multilingual
trainer = GLiNER2Trainer(model, training_config, compute_metrics=compute_metrics)

# %% [markdown]
# ## Train run

# %%
results = trainer.train(
    train_data=train_split, 
    eval_data=val_split
)

# %% [markdown]
# # Analysis

# %% [markdown]
# ## Training details

# %%
best_run = max(results["eval_metrics_history"], key=lambda x: x['f1'])
best_epoch = best_run['epoch']
best_p = best_run['precision']
best_r = best_run['recall']
best_f1 = best_run['f1']
total_epochs = len(results["eval_metrics_history"])
def get_cnt(data):
    exs = getattr(data, "examples", data)
    return sum(len(mentions) for ex in exs for mentions in ex.entities.values())

logger.info(f"Training completed!")
logger.info(f"Experiment name: {experiment_name}")
logger.info(f"Total steps: {results['total_steps']}")
logger.info(f"Total epochs: {total_epochs}")
logger.info(f"Training time: {results['total_time_seconds']/60:.1f} minutes")
logger.info(f"Best Epoch: {best_epoch + 1}/{total_epochs}") # +1 for 1-based indexing
logger.info(f"Best PRF: Precision: {best_p:.4f}, Recall: {best_r:.4f}, F1: {best_f1:.4f}")
# 2. Prepare data rows
table_data = [
    ["Train Split",  len(train_split),             get_cnt(train_split)],
    ["Val Split",    len(val_split),               get_cnt(val_split)],
    ["Full Dataset", len(train_dataset.examples), get_cnt(train_dataset)]
]
# 3. Print table
logger.info(tabulate(table_data, headers=["Subset", "Samples", "Mentions"], tablefmt="rounded_grid"))


# %% [markdown]
# ## Training progress

# %%
import matplotlib.pyplot as plt

# 1. Extract metrics from history
history = results['eval_metrics_history']
epochs = [h['epoch'] + 1 for h in history]
f1_scores = [h['f1'] for h in history]
precision = [h['precision'] for h in history]
recall = [h['recall'] for h in history]
losses = [h['eval_loss'] for h in history]

# 2. Setup the plot
plot_training_history(results)


# %% [markdown]
# ## Evaluation on dev using the best LoRA

# %%

# 1. Load the original base model (pristine weights)
best_model = GLiNER2.from_pretrained("fastino/gliner2-multi-v1")
# 2. Add the LoRA
adapter_path = DATA_DIR / "models" / experiment_name / "best"
best_model.load_adapter(adapter_path)
# 3. Ready for inference
logger.info("Adapter loaded.")


def evaluate_adapter(model, adapter_path, test_data, threshold=THRESHOLD):
    """
    Loads a specific LoRA adapter and evaluates its performance.
    """
    # 1. Load the specific weights
    logger.info(f"Loading adapter from: {adapter_path}")
    model.load_adapter(adapter_path)
    
    test_data = [
        (ex.text, {"entities": ex.entities, "entity_descriptions": ex.entity_descriptions}) 
        for ex in test_data
    ]

    # 2. Execute metric calculation
    results = compute_metrics(model, test_data, threshold=threshold)
    logger.info(f"\n--- EVALUATION RESULTS ({adapter_path.name}) threshold: {threshold} ---")
    logger.info(f"F1 Score : {results['f1']}")
    logger.info(f"Precision: {results['precision']}")
    logger.info(f"Recall   : {results['recall']}")
    logger.info(f"Counts   : TP={results['tp']}, FP={results['fp']}, FN={results['fn']}")
    
    return results

final_results = evaluate_adapter(best_model, adapter_path, val_split, threshold=THRESHOLD )


# %% [markdown]
# ## Threshold & Precision-Recall Analysis
#

# %%

# 1. Prepare data once
test_data_formatted = [
    (ex.text, {"entities": ex.entities, "entity_descriptions": ex.entity_descriptions}) 
    for ex in val_split
]

# 2. Iterate through thresholds
thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
p_scores = []
r_scores = []
f1_scores = []

logger.info("Analyzing Precision-Recall trade-off (N=21)")

for t in thresholds:
    print(f"\n[Threshold: {t:.2f}]", end=" ") 
    res = compute_metrics(best_model, test_data_formatted, threshold=t)
    p_scores.append(res['precision'])
    r_scores.append(res['recall'])
    f1_scores.append(res['f1'])

logger.info("Done analyzing Precision-Recall trade-off (N=21)")

# 3. Visualization
plot_threshold_curves(thresholds, p_scores, r_scores, f1_scores, THRESHOLD)

# %% [markdown]
# ## Evaluation on test set

# %%
DEFAULT_ANNOTATOR = env_vars.get("ANNOTATOR_A")
# Remove the redundant reload from ARGILLA_TEST_DATASET
# df_annotated = get_dataset_as_dataframe(...) was here

test_examples = df_to_gliner_examples(df_all, entity_descriptions)

logger.info(f"Text: {test_examples[20].text}")
logger.info(f"Entities: {test_examples[20].entities}")
logger.info(test_examples[20].entities)
logger.info(f'Entity descriptions example: ARTEFACT: {test_examples[20].entity_descriptions["ARTEFACT"]}')
logger.info(test_examples[20])


test_dataset = TrainingDataset(test_examples)
test_dataset.validate(raise_on_error=True)

for ds_name, ds in {"TEST": test_dataset}.items():
    logger.info(f"Dataset: {ds_name} ")
    ds.print_stats()
    logger.debug(ds[0])



# %%
# FINAL RESULTS BECOME FINAL RESULTS ON THE TEST SET!!!!!!!!!!!!!!!!!!!!!!!!

logger.info(f"Started evaluating model from {adapter_path}")
final_results = evaluate_adapter(best_model, adapter_path, test_dataset, threshold=0.8 )
logger.info(f"Done evaluating model from {adapter_path}")

# %%
# 1. Prepare data once
test_data_formatted = [
    (ex.text, {"entities": ex.entities, "entity_descriptions": ex.entity_descriptions}) 
    for ex in test_dataset # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
]


# %% [markdown]
# ## Confusion matrix

# %%
def plot_ner_confusion_matrix(model, dataset, threshold=THRESHOLD):
    y_true = []
    y_pred = []
    labels = list(entity_descriptions.keys())
    
    # 1. Collect all spans
    for ex in dataset:
        text, gt_entities = ex[0], ex[1]["entities"]
        
        # Ground Truth spans
        gt_spans = []
        for lbl, texts in gt_entities.items():
            for t in texts: gt_spans.append((t, lbl))
            
        # Prediction spans
        output = model.extract_entities(text, entity_descriptions, threshold=threshold)
        pred_entities = output.get('entities', {})
        pred_spans = []
        for lbl, texts in pred_entities.items():
            for t in texts: pred_spans.append((t, lbl))
            
        # 2. Match spans (Exact match logic)
        temp_pred = pred_spans.copy()
        for t_gt, lbl_gt in gt_spans:
            # Did we find this text/span?
            match = next((p for p in temp_pred if p[0] == t_gt), None)
            if match:
                y_true.append(lbl_gt)
                y_pred.append(match[1]) # record pred label (could be same or different)
                temp_pred.remove(match)
            else:
                y_true.append(lbl_gt)
                y_pred.append("O") # False Negative
        
        # Remaining predictions are False Positives
        for t_p, lbl_p in temp_pred:
            y_true.append("O")
            y_pred.append(lbl_p)

    # 3. Create Matrix
    all_labels = labels + ["O"]
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    
    # 4. Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=all_labels, yticklabels=all_labels, cmap='Blues')
    plt.title(f"NER Confusion Matrix (Threshold: {threshold})")
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

# Run it
plot_ner_confusion_matrix(best_model, test_data_formatted, threshold=THRESHOLD)


# %% [markdown]
# ### Confusion Matrix Interpretation Guide
#
# *   **Diagonal Cells**: True Positives (TP). Correct entity and correct label.
# *   **"O" Row (Bottom)**: False Positives (FP). The model hallucinated an entity where none existed.
# *   **"O" Column (Right)**: False Negatives (FN). The model completely missed a ground-truth entity.
# *   **Off-Diagonal (Non-"O")**: Label Misclassification. The model found the correct text span but assigned the wrong category (e.g., predicted `CONTEXT` for an `ARTEFACT`).
#

# %% [markdown]
#  ## Error Analysis 

# %%
def show_error_analysis(model, dataset, threshold=0.8, num_examples=5):
    print(f"--- QUALITATIVE ERROR ANALYSIS (Threshold: {threshold}) ---\n")
    
    for i, ex in enumerate(dataset[:num_examples]):
        text, gt_entities = ex[0], ex[1]["entities"]
        
        # 1. Get Predictions
        output = model.extract_entities(text, entity_descriptions, threshold=threshold)
        pred_entities = output.get('entities', {})
        
        # 2. Flatten for comparison
        gt_spans = [(t, lbl) for lbl, texts in gt_entities.items() for t in texts]
        pred_spans = [(t, lbl) for lbl, texts in pred_entities.items() for t in texts]
        
        # 3. Categorize
        tp = [p for p in pred_spans if p in gt_spans]
        fp = [p for p in pred_spans if p not in gt_spans]
        fn = [g for g in gt_spans if g not in pred_spans]
        
        # 4. Display
        print(f"EXAMPLE {i+1}:")
        print(f"TEXT: {text[:150]}...")
        
        if tp: print(f"  [TPs]: {tp}")
        if fp: print(f"  \033[91m[FPs]: {fp}\033[0m") # Red
        if fn: print(f"  \033[93m[FNs]: {fn}\033[0m") # Yellow
        print("-" * 50)

# Run on the first N examples
show_error_analysis(best_model, test_data_formatted, threshold=THRESHOLD, num_examples=50)


# %% [markdown]
# # Model saving if on Colab

# %%
if IN_COLAB:
    import shutil
    from google.colab import files
    # 1. Zip the adapter folder using pure Python
    # This creates 'best_model.zip' from the adapter_path folder
    shutil.make_archive("best_model", "zip", adapter_path)
    
    # 2. Trigger the browser download
    files.download("best_model.zip")
