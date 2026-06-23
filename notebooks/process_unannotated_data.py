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
# # Process Unannotated Data for Student Annotation
#
# This script loads unannotated raw text data, splits it into sentences, running GLiNER2 entity predictions, and pushes them to Argilla for student annotation.

# %%
import os
import json
import logging
from pathlib import Path
import regex as re
import itertools
import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from dotenv import dotenv_values, find_dotenv

from wtpsplit import SaT
from gliner2 import GLiNER2
import argilla as rg

from archaeo_ner_greek.utils import (
    get_argilla_client,
    setup_ner_dataset,
    add_records_safely
)

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)

# %% [markdown]
# ## Configuration

# %%
# --- HF Dataset Config ---
HF_REPO_ID = "Stalexan/archaeo-ner-greek"
HF_SUBSET = "raw_texts"
HF_SPLIT = "train"
USED_FOR_FILTER = "synthetic"

# --- Argilla Config ---
ARGILLA_DATASET_NAME = "archaeo_ner_greek-20260623"
SOURCE_DATASET_NAME = "archaeo_ner_greek"
ARGILLA_MIN_SUBMITTED = 1

# --- GLiNER2 Model Config ---
BASE_MODEL_NAME = "fastino/gliner2-multi-v1"
ADAPTER_NAME = "gliner2_archaeo_lora_20260520_1445"
INFERENCE_THRESHOLD = 0.8
SUGGESTION_AGENT_NAME = "GLiNER2_augmented_seeded_1to1"

# %% [markdown]
# ## 1. Load Environment & Dataset
# Fetching the configuration partition from Hugging Face and filtering for synthetic documents.

# %%
# Load environment vars
env_path = find_dotenv()
env_vars = dotenv_values(env_path) if env_path else {}
logger.info(f"Loaded environment variables from: {env_path}")

# --- Early Argilla Connectivity & Permission Checks ---
logger.info("Starting early Argilla connectivity and permission checks...")
try:
    client = get_argilla_client(env_vars=env_vars)
    workspace_name = env_vars.get("ARGILLA_WORKSPACE", "archaeo_ner_greek")
    
    # 1. Verify read permission on source dataset and retrieve settings
    logger.info(f"Checking read access on source dataset '{SOURCE_DATASET_NAME}'...")
    source_ds = client.datasets(name=SOURCE_DATASET_NAME, workspace=workspace_name)
    if not source_ds:
        raise ValueError(f"Source dataset '{SOURCE_DATASET_NAME}' not found in workspace '{workspace_name}'")
    logger.info("Successfully read source dataset schema.")
    
    # 2. Verify write permission & recreate empty target dataset immediately
    logger.info(f"Managing target dataset '{ARGILLA_DATASET_NAME}'...")
    existing_target = client.datasets(name=ARGILLA_DATASET_NAME, workspace=workspace_name)
    if existing_target:
        logger.info(f"Deleting existing target dataset: {ARGILLA_DATASET_NAME}")
        existing_target.delete()
        logger.info("Successfully deleted existing target dataset.")
        
    # Copy settings from source to target
    logger.info("Cloning schema settings to new target dataset...")
    source_settings = source_ds.settings
    target_settings = rg.Settings(
        fields=source_settings.fields,
        questions=source_settings.questions,
        metadata=source_settings.metadata,
        vectors=source_settings.vectors,
        guidelines=source_settings.guidelines,
        allow_extra_metadata=source_settings.allow_extra_metadata,
        distribution=rg.TaskDistribution(min_submitted=ARGILLA_MIN_SUBMITTED)
    )
    
    # Create empty dataset right away so it is visible in Argilla
    dataset = rg.Dataset(
        name=ARGILLA_DATASET_NAME,
        workspace=workspace_name,
        settings=target_settings,
        client=client
    )
    dataset.create()
    logger.info(f"✅ Successfully created empty target dataset '{ARGILLA_DATASET_NAME}' (visible in UI).")
except Exception as e:
    logger.critical(f"❌ Argilla connectivity/permission check failed: {e}")
    raise e

repo_id = env_vars.get("HF_REPO_ID", HF_REPO_ID)
hf_token = env_vars.get("HF_TOKEN") or env_vars.get("HUGGING_FACE_HUB_TOKEN")

# 1. Load HF dataset
logger.info(f"Loading {HF_SUBSET} from HF dataset: {repo_id}")
ds = load_dataset(repo_id, name=HF_SUBSET, split=HF_SPLIT, token=hf_token)
df_raw = ds.to_pandas()
df_hf = df_raw[df_raw["used_for"] == USED_FOR_FILTER].copy()
logger.info(f"Loaded {len(df_hf)} records from HF with used_for == '{USED_FOR_FILTER}'")

# 2. Load local files
extra_texts_dir = Path("data/extra_texts")
logger.info(f"Loading local texts from {extra_texts_dir}...")
local_records = []
if extra_texts_dir.exists():
    for filename in sorted(os.listdir(extra_texts_dir)):
        if filename.endswith(".txt") and "clarin" in filename and "tourkokratia" not in filename:
            filepath = extra_texts_dir / filename
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()
            local_records.append({
                "doc_id": filename,
                "raw_text": content
            })
else:
    logger.error(f"Directory {extra_texts_dir} does not exist.")

df_local = pd.DataFrame(local_records)
logger.info(f"Loaded {len(df_local)} local files matching '*clarin*' (excluding 'tourkokratia').")

# 3. Concatenate HF and local records
df_filtered = pd.concat([df_hf[["doc_id", "raw_text"]], df_local], ignore_index=True)
logger.info(f"Total records to process (HF + Local): {len(df_filtered)}")


# %% [markdown]
# ## 2. Sentence Splitting & Cleanup Heuristics

# %%
# Common abbreviation pattern compiler for Greek texts
ABBR_PATTERN = re.compile(
    r"\b(αι|αἱ|εικ|εἰκ|αρ|ἀρ|σελ|κεφ|κ\.λπ|κ\.λ\.π|χλμ|εκ|τ\.μ|\p{L})\.$", 
    re.IGNORECASE
)

LIST_ITEM_PATTERN = re.compile(r"^\s*\d+[\.\)-]\s*$")

def merge_list_fragments(sentences: list[str]) -> list[str]:
    """
    Merges sentences that are fragments of a numbered list.
    """
    merged_sentences = []
    for sentence in sentences:
        if not merged_sentences:
            merged_sentences.append(sentence)
            continue
            
        previous_sentence = merged_sentences[-1]
        
        if LIST_ITEM_PATTERN.match(previous_sentence):
            merged_sentences[-1] = f"{previous_sentence}{sentence}"
        else:
            merged_sentences.append(sentence)
            
    return merged_sentences

def merge_abbreviation_fragments(sentences: list[str]) -> list[str]:
    """
    Heuristically rejoins sentences that were split on common abbreviations (e.g. B.C., century, figures).
    """
    merged = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if not merged:
            merged.append(s)
            continue
            
        prev = merged[-1]
        # Check if previous segment ends with common abbreviation or single letter dot
        if ABBR_PATTERN.search(prev) or prev.endswith("π.Χ") or prev.endswith("μ.Χ") or prev.endswith("π.") or prev.endswith("μ."):
            merged[-1] = f"{prev} {s}"
        else:
            merged.append(s)
    return merged

# Initialize SaT Model
logger.info("Initializing SaT model (sat-3l, style=ud, lang=el)...")
sat = SaT("sat-3l", style_or_domain="ud", language="el")

# Process raw texts to sentences
processed_rows = []
for idx, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Splitting Sentences"):
    doc_id = row["doc_id"]
    raw_text = str(row["raw_text"])
    
    # Split text lines first to respect paragraph boundaries
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if not lines:
        continue
        
    # Split using SaT
    batch_results = sat.split(lines)
    sentences = list(itertools.chain.from_iterable(batch_results))
    
    # Apply post-processing merges
    sentences = merge_list_fragments(sentences)
    sentences = merge_abbreviation_fragments(sentences)
    
    for i, sent_text in enumerate(sentences):
        sent_text = sent_text.strip()
        if not sent_text:
            continue
            
        # Context extraction
        start_prev = max(0, i - 2)
        prev_text = " ".join(sentences[start_prev:i]) if i > 0 else "---"
        
        end_next = min(len(sentences), i + 3)
        next_text = " ".join(sentences[i+1:end_next]) if i < len(sentences) - 1 else "---"
        
        processed_rows.append({
            "doc_sent_id": f"{doc_id}_{i}",
            "doc_id": doc_id,
            "sentence": sent_text,
            "prev_context": prev_text,
            "next_context": next_text,
            "local_index": i
        })

df_sentences = pd.DataFrame(processed_rows)
logger.info(f"Generated {len(df_sentences)} sentences from {len(df_filtered)} documents.")

# %% [markdown]
# ## 3. GLiNER2 Model Inference & Prediction

# %%
# Load label definitions
base_dir = Path(__file__).resolve().parents[1]
labels_path = base_dir / "archaeo_ner_greek" / "resources" / "archaeoner_labels_definitions_v12_st.json"
logger.info(f"Loading label definitions from: {labels_path}")
with open(labels_path, "r", encoding="utf-8") as f:
    entity_descriptions = json.load(f)

# Load GLiNER2 and the fine-tuned adapter
logger.info("Loading GLiNER2 base model and fine-tuned adapter...")
model = GLiNER2.from_pretrained(BASE_MODEL_NAME)
adapter_path = base_dir / "data" / "models" / ADAPTER_NAME / "best"
logger.info(f"Loading adapter weights from: {adapter_path}")
model.load_adapter(str(adapter_path))
model.eval()

# Running predictions
logger.info("Running entity prediction/inference...")
predictions = []

for idx, row in tqdm(df_sentences.iterrows(), total=len(df_sentences), desc="Predicting Entities"):
    sentence_text = row["sentence"]
    try:
        output = model.extract_entities(sentence_text, entity_descriptions, threshold=INFERENCE_THRESHOLD)
        # Parse flat predictions format
        ents = output.get("entities", {})
        flat_preds = []
        for label, texts in ents.items():
            for t in texts:
                # Find character offsets in the sentence
                pattern = re.escape(t)
                for match in re.finditer(pattern, sentence_text):
                    flat_preds.append({
                        "label": label,
                        "start": match.start(),
                        "end": match.end(),
                        "text": t
                    })
        # Remove nested entities of same label
        flat_preds.sort(key=lambda x: (x["start"], -1 * (x["end"] - x["start"])))
        kept_preds = []
        for candidate in flat_preds:
            is_nested = False
            for existing in kept_preds:
                if candidate["start"] >= existing["start"] and candidate["end"] <= existing["end"]:
                    if candidate["label"] == existing["label"]:
                        is_nested = True
                        break
            if not is_nested:
                kept_preds.append(candidate)
        predictions.append(kept_preds)
    except Exception as e:
        logger.error(f"Inference failed for index {idx}: {e}")
        predictions.append([])

df_sentences["predictions"] = predictions

# %% [markdown]
# ## 4. Upload to Argilla Dataset

# %%
logger.info("Setting up Argilla connection and dataset...")
client = get_argilla_client(env_vars=env_vars)

workspace_name = env_vars.get("ARGILLA_WORKSPACE", "archaeo_ner_greek")

# Retrieve the already initialized target dataset
dataset = client.datasets(name=ARGILLA_DATASET_NAME, workspace=workspace_name)
if not dataset:
    raise ValueError(f"Target dataset '{ARGILLA_DATASET_NAME}' not found in workspace '{workspace_name}'")

target_settings = dataset.settings

# Get names of fields and metadata from the schema
fields_schema = [f.name for f in target_settings.fields]
metadata_schema = [m.name for m in target_settings.metadata] if target_settings.metadata else []

records = []
for idx, row in tqdm(df_sentences.iterrows(), total=len(df_sentences), desc="Building Argilla Records"):
    # Format suggestions for the SpanQuestion
    formatted_suggestions = []
    for pred in row["predictions"]:
        formatted_suggestions.append({
            "label": str(pred["label"]),
            "start": int(pred["start"]),
            "end": int(pred["end"]),
            "score": 1.0
        })
        
    # Dynamically map record fields based on target schema
    fields = {}
    if "sentence_field" in fields_schema:
        fields["sentence_field"] = str(row["sentence"])
    if "prev_sentences_field" in fields_schema:
        fields["prev_sentences_field"] = str(row["prev_context"])
    if "next_sentences_field" in fields_schema:
        fields["next_sentences_field"] = str(row["next_context"])
    if "document_sentence_id_field" in fields_schema:
        fields["document_sentence_id_field"] = str(row["doc_sent_id"])

    # Fallback mappings
    for f_name in fields_schema:
        if f_name not in fields:
            if f_name in ("sentence", "text"):
                fields[f_name] = str(row["sentence"])
            else:
                fields[f_name] = "---"

    # Dynamically map metadata based on target schema
    metadata = {}
    if "context_sheet_description_id" in metadata_schema:
        metadata["context_sheet_description_id"] = int(idx)
    if "sentence_id_metadata" in metadata_schema:
        metadata["sentence_id_metadata"] = int(idx)
    if "doc_id" in metadata_schema:
        metadata["doc_id"] = str(row["doc_id"])
    if "local_index" in metadata_schema:
        metadata["local_index"] = int(row["local_index"])

    record = rg.Record(
        fields=fields,
        metadata=metadata,
        suggestions=[
            rg.Suggestion(
                question_name="entities",
                value=formatted_suggestions,
                agent=SUGGESTION_AGENT_NAME
            )
        ],
        id=str(row["doc_sent_id"])
    )
    records.append(record)

logger.info(f"Logging {len(records)} records to Argilla dataset: {ARGILLA_DATASET_NAME}")
add_records_safely(client, ARGILLA_DATASET_NAME, workspace_name, records)
logger.info("Successfully uploaded unannotated predictions to Argilla!")

# %%
