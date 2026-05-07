# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# # Fine-tuning GLiNER2 for Archaeological Named Entity Recognition
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
# ## Environment Initialization & Configuration

# %%
import os
import sys
import subprocess

# --- STANDALONE BOOTSTRAP FOR COLAB ---
# Clones the repo and sets paths BEFORE internal package imports.
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    print("Pipeline Version: 1.2.3")
    try:
        from google.colab import userdata
        GITHUB_TOKEN = userdata.get('GITHUB_TOKEN')
        REPO_NAME = "archaeo-ner-greek"
        if not os.path.exists(REPO_NAME):
            REPO_URL = f"https://{GITHUB_TOKEN}@github.com/Stav788/{REPO_NAME}.git"
            subprocess.check_call(["git", "clone", "--branch", "dev", REPO_URL])
        
        if os.path.abspath(REPO_NAME) not in sys.path:
            sys.path.append(os.path.abspath(REPO_NAME))
        os.chdir(REPO_NAME)
    except Exception as e:
        print(f"Colab bootstrap failed: {e}")

# %%
import json
import logging
import warnings
import random
from datetime import datetime
from logging.config import dictConfig
from pathlib import Path

# Try to import wandb (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from archaeo_ner_greek.training_utils import (
    setup_local, setup_colab, df_to_gliner_examples, verify_annotations, 
    plot_training_history, plot_threshold_curves, extract_doc_ids, grouped_split, 
    plot_ner_confusion_matrix, compute_metrics, get_cnt, evaluate_adapter, 
    show_error_analysis, find_best_split_seeds, show_detailed_report,
    safe_wandb_log, setup_wandb, upload_wandb_artifact,
    DEFAULT_ANNOTATOR, VERIFICATION_TARGET_IDS, VERIFICATION_TARGETS, 
    VERIFICATION_PAIR_TEXT, VERIFICATION_PAIR_LABEL
)

# 1. Environment Detection & Pre-Import Setup
if IN_COLAB:
    env_vars = setup_colab()
else:
    env_vars = setup_local()

# 2. Optimized Imports
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
from archaeo_ner_greek.logging_config import setup_logging
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
log_file = setup_logging()
logger = logging.getLogger(__name__)
logger.info(f">>> Logging to: {log_file}")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings(action="ignore", message=r"datetime.datetime.utcnow")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*SwigPyObject.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="wandb")

logger.info(f">>> Working Directory: {BASE_DIR}")
logger.info(f">>> Models Directory:  {MODELS_DIR}")
logger.info(f">>> Dataset:  {env_vars['ARGILLA_DATASET']}")
logger.info(f">>> Test dataset: {env_vars['ARGILLA_TEST_DATASET']}")

# WandB Status for later logging
wandb_enabled = WANDB_AVAILABLE and env_vars.get("WANDB_API_KEY") is not None

# %% [markdown]
# ## Data Loading and Preprocessing
#

# %%
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

# Data is already filtered by get_dataset_as_dataframe(username=DEFAULT_ANNOTATOR)
logger.info(f"Using data for annotator: {DEFAULT_ANNOTATOR}")

# Extract Parent Document IDs
df_all['doc_id'] = extract_doc_ids(df_all)
logger.info(f"Unique documents identified: {df_all['doc_id'].nunique()}")
logger.info("\n>>> DOC_ID MAPPING SAMPLE")
logger.info(df_all[['document_sentence_id_field', 'doc_id']].head(10))


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
# ### Semantic Schema & Entity Label Definitions 

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
verify_annotations(
    df_all, 
    VERIFICATION_TARGET_IDS, 
    VERIFICATION_TARGETS, 
    VERIFICATION_PAIR_TEXT, 
    VERIFICATION_PAIR_LABEL, 
    DEFAULT_ANNOTATOR
)

# %% [markdown]
# ### Document-Aware Stratified Partitioning (80/10/10)

# %%
# Find the most balanced split across 100 random seeds
logger.info("Searching for the most balanced document split (Best-of-100)...")
best_seeds = find_best_split_seeds(df_all, group_col='doc_id', num_trials=100, top_n=5)

# Log the top candidates for transparency
for i, candidate in enumerate(best_seeds):
    counts = candidate['counts']
    logger.info(f"Top {i+1}: Seed {candidate['seed']} | Split Counts: Train={counts[0]}, Val={counts[1]}, Test={counts[2]}")

# Select the absolute best seed
BEST_SEED = best_seeds[0]['seed']
logger.info(f"Selecting BEST_SEED={BEST_SEED} for this experiment.")

# Create the final grouped splits
df_train, df_val, df_test = grouped_split(df_all, group_col='doc_id', seed=BEST_SEED)
logger.info(f"Final Split Ratios: Train={len(df_train)/len(df_all):.1%}, Val={len(df_val)/len(df_all):.1%}, Test={len(df_test)/len(df_all):.1%}")

# %% [markdown]
# ### Dataset Serialization for GLiNER2

# %%
train_examples = df_to_gliner_examples(df_train, entity_descriptions)
val_examples   = df_to_gliner_examples(df_val, entity_descriptions)
test_examples  = df_to_gliner_examples(df_test, entity_descriptions)

logger.info(f"Text Sample: {train_examples[0].text}")
logger.info(f"Entities Sample: {train_examples[0].entities}")




# %%

train_split = TrainingDataset(train_examples)
val_split   = TrainingDataset(val_examples)
test_split  = TrainingDataset(test_examples) # Isolated Gold set for final benchmark

logger.info(f"Grouped Stats: Train={len(train_split)} | Val={len(val_split)} | Test={len(test_split)}")


for ds_name, ds in {"train": train_split, "val": val_split, "test": test_split}.items():
    logger.info(f"Dataset: {ds_name} ")
    ds.print_stats()
    logger.debug(ds[0])

# %% [markdown]
# # Model Optimization & Fine-tuning


# %% [markdown]
# ## Hyperparameter & Architecture Configuration

# %%
experiment_name = f"gliner2_archaeo_lora_{datetime.now().strftime('%Y%m%d_%H%M')}"
output_dir = DATA_DIR / "models" / experiment_name
output_dir.mkdir(parents=True, exist_ok=True)

# Save Split Manifest for reproducibility
split_manifest = {
    "train": sorted(df_train['doc_id'].unique().tolist()),
    "val": sorted(df_val['doc_id'].unique().tolist()),
    "test": sorted(df_test['doc_id'].unique().tolist()),
    "stats": {
        "train_samples": len(df_train),
        "val_samples": len(df_val),
        "test_samples": len(df_test)
    },
    "seed": BEST_SEED
}
with open(output_dir / "split_manifest.json", "w") as f:
    json.dump(split_manifest, f, indent=4)
logger.info(f"Split Manifest saved to {output_dir}/split_manifest.json")
num_epochs = 20


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


    # Evaluation & Logging
    eval_strategy="epoch",         # Saves best checkpoint at end of every epoch
    save_best=True,
    report_to_wandb=wandb_enabled,
    wandb_project=env_vars.get("WANDB_PROJECT", "archaeo-ner-greek"),
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

# Initialize WandB run & capture logging function
log_to_wandb = setup_wandb(wandb_enabled, training_config.wandb_project, experiment_name, training_config)

# %% [markdown]
# ## Base Model Instantiation

# %%
model = GLiNER2.from_pretrained("fastino/gliner2-multi-v1")

# %% [markdown]
# ## Training Engine Setup (GLiNER2Trainer)

# %%
trainer = GLiNER2Trainer(model, training_config, compute_metrics=compute_metrics)

# %% [markdown]
# ## Execution of Fine-tuning Pipeline

# %%
results = trainer.train(
    train_data=train_split, 
    eval_data=val_split
)

# %% [markdown]
# # Evaluation & Error Diagnostics

# %% [markdown]
# ## Post-Training Session Summary

# %%
best_run = max(results["eval_metrics_history"], key=lambda x: x['f1'])
best_epoch = best_run['epoch']
best_p = best_run['precision']
best_r = best_run['recall']
best_f1 = best_run['f1']
total_epochs = len(results["eval_metrics_history"])

logger.info(f"Training completed!")
logger.info(f"Experiment name: {experiment_name}")
logger.info(f"Total steps: {results['total_steps']}")
logger.info(f"Total epochs: {total_epochs}")
logger.info(f"Training time: {results['total_time_seconds']/60:.1f} minutes")
logger.info(f"Best Epoch: {best_epoch + 1}/{total_epochs}") # +1 for 1-based indexing
logger.info(f"Best PRF: Precision: {best_p:.4f}, Recall: {best_r:.4f}, F1: {best_f1:.4f}")
# 2. Prepare data rows
table_data = [
    ["Train Split",   len(train_split), get_cnt(train_split)],
    ["Val Split",     len(val_split),   get_cnt(val_split)],
    ["Gold Test Set", len(test_split),  get_cnt(test_split)]
]
# 3. Print table
print(tabulate(table_data, headers=["Subset", "Samples", "Mentions"], tablefmt="grid"))


# %% [markdown]
# ## Convergence Analysis & Loss Visualization

# %%
# Training history visualization

# 2. Setup the plot
plot_training_history(results)


# %% [markdown]
# ## Model Validation & Checkpoint Selection (Dev Set)

# %%

# 1. Load the original base model (pristine weights)
best_model = GLiNER2.from_pretrained("fastino/gliner2-multi-v1")
# 2. Add the LoRA
adapter_path = DATA_DIR / "models" / experiment_name / "best"
best_model.load_adapter(adapter_path)
# 3. Ready for inference
logger.info("Adapter loaded.")

# %% [markdown]
# ## Inference Threshold Calibration
# We determine the optimal confidence threshold using the Validation Set to maximize the F1-score.

# %%
# 1. Prepare data once
val_data_formatted = [
    (ex.text, {"entities": ex.entities, "entity_descriptions": ex.entity_descriptions}) 
    for ex in val_split
]

# 2. Iterate through thresholds to find the best F1
thresholds = [0.4, 0.5, 0.6, 0.7, 0.8]
p_scores = []
r_scores = []
f1_scores = []

best_threshold = 0.5
best_val_f1 = 0

logger.info(f"Finding optimal threshold on validation set ({len(val_split)} samples)...")

for t in thresholds:
    logger.info(f"Evaluating threshold: {t}...")
    res = compute_metrics(best_model, val_data_formatted, threshold=t)
    p_scores.append(res['precision'])
    r_scores.append(res['recall'])
    f1_scores.append(res['f1'])
    
    if res['f1'] > best_val_f1:
        best_val_f1 = res['f1']
        best_threshold = t

logger.info(f">>> Optimal Threshold Found: {best_threshold} (Val F1: {best_val_f1:.4f})")

# 3. Visualization of the search space
plot_threshold_curves(thresholds, p_scores, r_scores, f1_scores, best_threshold)

# %% [markdown]
# ## Zero-Shot Performance Baseline
# Comparative evaluation of the base model's zero-shot capabilities prior to domain adaptation.

# %%
logger.info("Evaluating Zero-Shot Baseline...")
# Load fresh base model without adapters
zero_shot_model = GLiNER2.from_pretrained("fastino/gliner2-multi-v1")
zero_shot_model.to(best_model.device)

# Prepare test data
test_data_formatted = [
    (ex.text, {"entities": ex.entities, "entity_descriptions": ex.entity_descriptions}) 
    for ex in test_split
]

# Evaluate at a standard threshold (0.5)
zs_results = compute_metrics(zero_shot_model, test_data_formatted, threshold=0.5)

logger.info("\n" + "="*40)
logger.info("ZERO-SHOT BASELINE RESULTS")
logger.info(f"F1 Score : {zs_results['f1']:.4f}")
logger.info(f"Precision: {zs_results['precision']:.4f}")
logger.info(f"Recall   : {zs_results['recall']:.4f}")
logger.info("="*40)

# Log to WandB
log_to_wandb(zs_results, prefix="zero_shot_")

# %% [markdown]
# ## Final Benchmarking on Isolated Gold Test Set
# Evaluation of the optimized adapter using the calibrated inference threshold.

# %%
# Use the isolated test set created during the grouped split
test_dataset = test_split
test_data_formatted = [
    (ex.text, {"entities": ex.entities, "entity_descriptions": ex.entity_descriptions}) 
    for ex in test_dataset
]

logger.info(f"Started evaluating model on GOLD TEST SET using optimal threshold: {best_threshold}")
final_results = evaluate_adapter(best_model, adapter_path, test_dataset, threshold=best_threshold)
logger.info(f"Done evaluating model.")

# Log final benchmark to WandB
log_to_wandb(final_results, prefix="gold_test_")

# Upload Best Adapter as Artifact
upload_wandb_artifact(wandb_enabled, adapter_path, experiment_name)

if wandb_enabled:
    wandb.finish()
# %% [markdown]
# ## Granular Performance Metrics by Entity Class
# Precision, Recall, and F1-score disaggregated by archaeological entity types.

# %%
show_detailed_report(best_model, test_data_formatted, threshold=best_threshold)

# %% [markdown]
# ## Categorical Confusion Analysis

# %%
# Plot confusion matrix using the optimized threshold
plot_ner_confusion_matrix(best_model, test_data_formatted, entity_descriptions, threshold=best_threshold)



# %% [markdown]
# ### Confusion Matrix Interpretation Guide
#
# *   **Diagonal Cells**: True Positives (TP). Correct entity and correct label.
# *   **"O" Row (Bottom)**: False Positives (FP). The model hallucinated an entity where none existed.
# *   **"O" Column (Right)**: False Negatives (FN). The model completely missed a ground-truth entity.
# *   **Off-Diagonal (Non-"O")**: Label Misclassification. The model found the correct text span but assigned the wrong category (e.g., predicted `CONTEXT` for an `ARTEFACT`).
#

# Run error analysis using the optimized threshold
show_error_analysis(best_model, test_data_formatted, entity_descriptions, threshold=best_threshold, num_examples=50)


# %% [markdown]
# Final evaluation completed. Best model adapter is persisted locally and uploaded as a WandB artifact.

# %% [markdown]
# # Usage & Documentation Guide
#
# This pipeline is designed for both local execution and cloud-based experimentation via Colab.
#
# ### 1. Local Environment Configuration
# To maintain a stable and reproducible environment, we use `uv` package manager.
# *   **Synchronization**: Ensure all dependencies and the local package are installed by running `uv sync` in the root directory.
# *   **Notebook Serialization**: If you are working directly with the `.py` source, you can regenerate the interactive notebook using Jupytext:
#     ```bash
#     uv run jupytext --to ipynb notebooks/gliner2_training.py
#     ```
# *   **Execution**: The script can be executed via VS Code, Antigravity, or any Jupyter-compatible IDE. Ensure the `.venv` kernel is selected.
#
# ### 2. Colab Integration
# The script features automatic environment detection and will trigger `setup_colab()` when running in the cloud.
# *   **Secrets Management**: Before execution, populate the following keys in the Colab "Secrets" sidebar:
#     *   `GITHUB_TOKEN`: Required for cloning the private repository.
#     *   `ARGILLA_API_URL` & `ARGILLA_API_KEY`: For corpus acquisition.
#     *   `ARGILLA_WORKSPACE`, `ARGILLA_DATASET`, `ARGILLA_TEST_DATASET`.
#     *   `ANNOTATOR_A`: Specifically identifies the annotator for data filtering.
#     *   `WANDB_API_KEY`: Enables remote experiment tracking.
# *   **Hardware Selection**: For fine-tuning performance, select a GPU or TPU runtime.
#
# ### 3. Experiment Tracking (Weights & Biases)
# This project leverages WandB for real-time experiment tracking and artifact persistence.
# *   **Metrics**: Training loss, validation metrics, and convergence curves are synchronized automatically.
# *   **Artifact Persistence**: The optimized LoRA adapter is uploaded as a versioned artifact at the conclusion of the run. This provides a cloud-based backup and ensures the best model state is never lost.
# *   **Comparative Baseline**: The dashboard includes both zero-shot and fine-tuned results to facilitate objective performance analysis.
#
# ### 4. Reproducibility & Artifacts
# *   **Local Storage**: Check the `data/models/{experiment_name}` directory for the best-performing adapter and local manifests.
# *   **Split Integrity (Leakage Prevention)**: The `split_manifest.json` preserves the exact document-grouped partitions used during the session. In archaeological texts, sentences from the same document often share localized terminology and context. Splitting a single document across partitions would cause "data leakage," leading to inflated, unrealistic performance metrics. Maintaining this manifest ensures that the evaluation is conducted on entirely "unseen" documents. This provides an audit trail ensuring the model's performance is based on learning rather than memorization on specific document.
