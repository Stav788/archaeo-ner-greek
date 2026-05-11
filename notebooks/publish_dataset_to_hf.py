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
# # Guidelines for Publishing Archaeological NER Datasets to Hugging Face
#
# This guide outlines the steps for a user to publish the consolidated dataset to their own Hugging Face (HF) account using the provided pipeline.
#
# ## 1. Prerequisites & Authentication
#
# - **HF Write Token**: Generate a token with **write** permissions at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
# - **Environment Variables**: Add the token to your local `.env` file as `HF_TOKEN`.
#
# ## 2. Configuration
#
# Ensure your `.env` contains the following target repository information:
#
# - `HF_REPO_ID`: The full identifier for your dataset (e.g., `your-username/archaeo-ner-greek`).
# - `HF_REPO_PRIVATE`: Set to `True` for private hosting, `False` for public.
# - `HF_REPO_GATED`: Set to `manual` to require approval for access, or `false` to disable gating.
# - `DEFAULT_ANNOTATOR`: Your specific Argilla username (the script will only publish your verified records).
#
# ## 3. Execution
#
# Run the consolidated publication script from the project root:
#
# ```bash
# uv run notebooks/publish_dataset_to_hf.py
# ```
#
# ## 4. Dataset Architecture (Automated)
#
# The script automatically produces two subsets to satisfy both training and archival needs:
#
# - **`argilla` subset**: A full archival backup containing every original record, column, and metadata field.
# - **`default` subset**: A clean, stratified training set with:
#   - **GLiNER2 Compatibility**: Standardized `input` (text) and `output` (entity dictionary) columns.
#   - **Stratified Splits**: Grouped by document ID (80/10/10) to ensure rare labels appear in every partition without data leakage.
#   - **Precision Preservation**: Original character-level offsets are kept in the `labels` column for auditing.
#
# ## 5. Verification & Governance
#
# - **Integrity Check**: The script performs an automatic "round-trip" verification, pulling the data back from HF and comparing record IDs and counts against the in-memory source.
# - **Access Control**: If `HF_REPO_GATED` is set, the script automatically configures the repository settings on the Hub.
# - **Contact Metadata**: The script updates the DatasetCard (`README.md`) with the `HF_NOTIFICATION_EMAIL` specified in your `.env`.

# %% [markdown]
# # Archaeological NER: Consolidated Dataset Publication
#
# This script standardizes the publication pipeline for the Archaeological NER project. 
# It automates the transition from raw Argilla annotations to a production-ready 
# Hugging Face dataset.
#
# ## Pipeline Stages:
# 1. **Data Acquisition**: Fetches records for DEFAULT_ANNOTATOR from multiple Argilla datasets.
# 2. **Deduplication**: Resolves semantic duplicates in-memory using NFKD normalization.
# 3. **Stratification**: Performs a document-grouped split (80/10/10) that ensures 
#    100% label representation in every partition.
# 4. **Multi-Subset Export**:
#    - **`argilla`**: Full archival backup containing all original columns and metadata.
#    - **`default`**: GLiNER2-formatted partitions (input/output) with offsets preserved.
# 5. **Governance**: Configures repository gating and metadata contact information.
# 6. **Verification**: Performs a round-trip count and schema validation.

# %%
import logging
import os
import sys
import pandas as pd
import argilla as rg
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from datasets import load_dataset

# Local project imports
from archaeo_ner_greek.utils import (
    get_argilla_client,
    connect_hugging_face,
    merge_datasets_in_memory
)

# --- Enhanced Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def log_stage(title):
    logger.info(f"\n{'='*20} STAGE: {title} {'='*20}")

# %% [markdown]
# ## 1. Initialization & Credentials
# Load environment variables and authenticate with services.

# %%
log_stage("INITIALIZATION")
# Load environment variables
env_path = find_dotenv()
load_dotenv(env_path, override=True)

# 1. Connect to Argilla
logger.info("Connecting to Argilla...")
try:
    client = get_argilla_client()
    workspace = os.getenv("ARGILLA_WORKSPACE", "archaeo_ner_greek")
except Exception as e:
    logger.error(f"Failed to connect to Argilla: {e}")
    sys.exit(1)

# 2. Connect to Hugging Face
logger.info("Connecting to Hugging Face Hub...")
try:
    hf_api = connect_hugging_face()
    repo_id = os.getenv("HF_REPO_ID")
    # Check both standard names for the token
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    hf_private = os.getenv("HF_REPO_PRIVATE", "True").lower() == "true"
    hf_gated = os.getenv("HF_REPO_GATED", "manual").lower()
    hf_notification_email = os.getenv("HF_NOTIFICATION_EMAIL")
except Exception as e:
    logger.error(f"Failed to connect to Hugging Face: {e}")
    sys.exit(1)

if not repo_id:
    logger.error("HF_REPO_ID not found in .env file.")
    sys.exit(1)

logger.info(f"Argilla Workspace: {workspace}")
logger.info(f"HF Repo ID: {repo_id}")
logger.info(f"HF Privacy: {'Private' if hf_private else 'Public'}")

# %% [markdown]
# ## 2. Merging Datasets (In-Memory)
# We pull `ARGILLA_DATASET` and `ARGILLA_TEST_DATASET` into RAM and combine them.
# **Note**: This process is read-only for the Argilla server.

# %%
log_stage("DATASET ACQUISITION & MERGING")
dataset_name = os.getenv("ARGILLA_DATASET")
test_dataset_name = os.getenv("ARGILLA_TEST_DATASET")

if not dataset_name or not test_dataset_name:
    logger.error("ARGILLA_DATASET or ARGILLA_TEST_DATASET not found in .env.")
    sys.exit(1)

logger.info(f"Fetching and merging '{dataset_name}' and '{test_dataset_name}' in memory...")

try:
    # Perform the merge using HF datasets concatenation
    merged_hf_ds = merge_datasets_in_memory(
        client=client,
        dataset_names=[dataset_name, test_dataset_name],
        workspace=workspace,
        username=os.getenv("DEFAULT_ANNOTATOR")
    )
except Exception as e:
    logger.error(f"Failed to merge datasets: {e}")
    sys.exit(1)

logger.info(f"Consolidation complete. In-memory pool contains {len(merged_hf_ds)} records.")

# %%
log_stage("STRATIFIED SPLITTING")
from archaeo_ner_greek.training_utils import find_best_split_seeds, grouped_split, extract_doc_ids

# 1. Convert merged dataset to a DataFrame for splitting
df_all = merged_hf_ds.to_pandas()
df_all['doc_id'] = extract_doc_ids(df_all)

# 2. Find best seed and split
logger.info("Calculating optimal stratified split...")
best_seeds = find_best_split_seeds(df_all, group_col='doc_id', num_trials=100, top_n=1)
best_seed = best_seeds[0]['seed']
logger.info(f"Selected Seed {best_seed} for stratification.")

df_train, df_val, df_test = grouped_split(df_all, group_col='doc_id', seed=best_seed)

# 3. Convert back to HF Datasets with Pruning for 'default' subset
from datasets import Dataset as HFDatasetDict
from datasets import Features, Value, Sequence

def prepare_gliner2_format(df, all_possible_labels):
    """
    Adds 'input' and 'output' columns matching GLiNER2 tutorial 
    while preserving 'labels' (offsets) and 'doc_id'.
    Ensures 'output' has a consistent schema by including all possible labels.
    """
    df = df.copy()
    df['input'] = df['sentence_field']
    
    # Create the 'output' column expected by GLiNER2
    outputs = []
    for idx, row in df.iterrows():
        text = row['sentence_field']
        # Initialize with all labels to ensure schema consistency
        entities = {lbl: [] for lbl in all_possible_labels}
        
        for ent in row['labels']:
            lbl = ent['label']
            mention = text[ent['start']:ent['end']].strip()
            if lbl in entities:
                entities[lbl].append(mention)
        outputs.append({"entities": entities})
    
    df['output'] = outputs
    
    # Final selection of columns
    cols_to_keep = ['input', 'output', 'labels', 'doc_id']
    return df[cols_to_keep]

# Get the master list of all labels present in the dataset
all_labels = set()
for labels_list in df_all['labels']:
    for ent in labels_list:
        all_labels.add(ent['label'])
all_labels = sorted(list(all_labels))
logger.info(f"Master Label Set for Schema: {all_labels}")

# Define EXPLICIT features to prevent schema mismatch errors during push
# This forces the 'output' column to have the same structure in all splits
# Note: Using Sequence(Features(...)) for labels to avoid "struct of lists" interpretation
features = Features({
    'input': Value('string'),
    'output': {
        'entities': {lbl: Sequence(Value('string')) for lbl in all_labels}
    },
    'labels': Sequence(Features({
        'label': Value('string'),
        'start': Value('int64'),
        'end': Value('int64')
    })),
    'doc_id': Value('string')
})

train_ds = HFDatasetDict.from_pandas(prepare_gliner2_format(df_train, all_labels), features=features, preserve_index=False)
val_ds = HFDatasetDict.from_pandas(prepare_gliner2_format(df_val, all_labels), features=features, preserve_index=False)
test_ds = HFDatasetDict.from_pandas(prepare_gliner2_format(df_test, all_labels), features=features, preserve_index=False)

# %% [markdown]
# ## 3. Push to Hugging Face Hub
# Export the consolidated dataset to the Hub.

# %%
log_stage("PUSH TO HUGGING FACE")
logger.info(f"Pushing dataset to Hugging Face Hub: {repo_id}...")

try:
    # A) Push the Full "argilla" subset (COMPLETE BACKUP)
    logger.info("Pushing 'argilla' subset (Full Backup with all columns)...")
    merged_hf_ds.push_to_hub(
        repo_id=repo_id,
        config_name="argilla",
        split="train",
        token=hf_token,
        private=hf_private
    )

    # B) Push the Partitioned "default" subset (CLEAN TRAINING DATA)
    logger.info("Pushing 'default' subset (Cleaned Train/Val/Test splits)...")
    from datasets import DatasetDict
    final_dd = DatasetDict({
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds
    })
    final_dd.push_to_hub(
        repo_id=repo_id,
        token=hf_token,
        private=hf_private
    )
    
    hf_url = f"https://huggingface.co/datasets/{repo_id}"
    logger.info(f"✅ SUCCESS: Published both subsets to: {hf_url}")
    
    # --- New: Apply Gating (Configurable) ---
    if hf_gated != "false":
        logger.info(f"Setting dataset gating to '{hf_gated}'...")
        hf_api.update_repo_settings(
            repo_id=repo_id,
            repo_type="dataset",
            gated=hf_gated
        )
        logger.info(f"✅ SUCCESS: Gating is now enabled with '{hf_gated}' approval.")
    else:
        logger.info("HF_REPO_GATED is 'false'. Skipping gating setup.")

    # --- New: Set Notification Email in Metadata ---
    if hf_notification_email:
        logger.info(f"Setting notification email to: {hf_notification_email}")
        from huggingface_hub import DatasetCard
        try:
            card = DatasetCard.load(repo_id, repo_type="dataset", token=hf_token)
            card.data.contact_email = hf_notification_email
            card.push_to_hub(repo_id, repo_type="dataset", token=hf_token)
            logger.info("✅ SUCCESS: Notification email added to metadata.")
        except Exception as card_err:
            logger.warning(f"Could not update DatasetCard: {card_err}. Creating a new one.")
            card = DatasetCard(f"---\ncontact_email: {hf_notification_email}\n---")
            card.push_to_hub(repo_id, repo_type="dataset", token=hf_token)
            logger.info("✅ SUCCESS: New DatasetCard created with notification email.")
    else:
        logger.warning("HF_NOTIFICATION_EMAIL not found in .env. Skipping metadata update.")
except Exception as e:
    logger.error(f"Failed to push to Hugging Face Hub: {e}")
    sys.exit(1)

# %% [markdown]
# ## 4. Verification Round-Trip
# Pull the dataset back from Hugging Face and compare it with the in-memory version to ensure integrity.

# %%
log_stage("VERIFICATION ROUND-TRIP")
logger.info("Pulling dataset back from Hugging Face for verification...")

try:
    # 1. Pull back the 'argilla' subset (Full Pool)
    pulled_argilla = load_dataset(repo_id, name="argilla", split="train", token=hf_token, download_mode="force_redownload")
    logger.info(f"Pulled {len(pulled_argilla)} records from 'argilla' subset.")
    
    # 2. Pull back the 'default' subset (Partitions)
    pulled_partitions = load_dataset(repo_id, name="default", token=hf_token, download_mode="force_redownload")
    logger.info(f"Pulled partitions: Train={len(pulled_partitions['train'])}, Val={len(pulled_partitions['validation'])}, Test={len(pulled_partitions['test'])}")
    
    # 3. Compare record counts
    original_count = len(merged_hf_ds)
    argilla_count = len(pulled_argilla)
    partitions_total = len(pulled_partitions['train']) + len(pulled_partitions['validation']) + len(pulled_partitions['test'])
    
    if original_count == argilla_count == partitions_total:
        logger.info(f"✅ SUCCESS: Record counts match exactly across all subsets ({original_count}).")
    else:
        logger.error(f"❌ FAILURE: Record count mismatch!")
        logger.error(f"   - Original in-memory: {original_count}")
        logger.error(f"   - 'argilla' subset:   {argilla_count}")
        logger.error(f"   - Sum of partitions:  {partitions_total}")
        sys.exit(1)

    # 4. Verify IDs (on the argilla subset)
    if 'id' in merged_hf_ds.column_names and 'id' in pulled_argilla.column_names:
        logger.info("Verifying individual record IDs in 'argilla' subset...")
        orig_ids = sorted(merged_hf_ds['id'])
        pulled_ids = sorted(pulled_argilla['id'])
        
        if orig_ids == pulled_ids:
            logger.info(f"✅ SUCCESS: All {original_count} record IDs are identical.")
        else:
            logger.error("❌ FAILURE: Record ID sets differ.")
            sys.exit(1)
            
    # 5. Verify GLiNER2 columns in 'default' subset
    train_split = pulled_partitions['train']
    if 'input' in train_split.column_names and 'output' in train_split.column_names:
        logger.info("✅ SUCCESS: 'input' and 'output' columns (GLiNER2 format) are present.")
    else:
        logger.error(f"❌ FAILURE: GLiNER2 columns missing in 'default' subset. Columns: {train_split.column_names}")
        sys.exit(1)
            
except Exception as e:
    logger.error(f"Verification process failed: {e}")
    sys.exit(1)

# %% [markdown]
# ## 5. Conclusion
# Execution completed.
log_stage("COMPLETED")
logger.info("The Argilla server remains untouched. Data is now synced to Hugging Face.")
