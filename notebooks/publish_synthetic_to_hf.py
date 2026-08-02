#!/usr/bin/env python3
import os
import sys
import logging
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from datasets import Dataset, DatasetDict, Features, Value, Sequence, load_dataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from archaeo_ner_greek.training_utils import setup_local, curate_synthetic_data

import re

def resolve_provenance(text, files_content):
    """Matches a synthetic sentence back to its source filename."""
    norm_text = re.sub(r'\s+', '', text).lower()
    for filename, content in files_content.items():
        norm_content = re.sub(r'\s+', '', content).lower()
        if norm_text in norm_content or norm_content in norm_text:
            return filename
            
    # Fallback Signature Match
    sig = text[:40].strip()
    norm_sig = re.sub(r'\s+', '', sig).lower()
    if len(norm_sig) >= 15:
        for filename, content in files_content.items():
            norm_content = re.sub(r'\s+', '', content).lower()
            if norm_sig in norm_content:
                return filename
                
    return "synthetic_generation"

def prepare_gliner2_format(df, all_possible_labels, files_content=None):
    """Formats synthetic DataFrame to match GLiNER2 expectations and maps provenance."""
    df = df.copy()
    if 'input' not in df.columns and 'sentence_field' in df.columns:
        df['input'] = df['sentence_field']
    
    outputs = []
    doc_ids = []
    for idx, row in df.iterrows():
        text = row['input']
        entities = {lbl: [] for lbl in all_possible_labels}
        for ent in row.get('labels', []):
            lbl = ent['label']
            mention = text[ent['start']:ent['end']].strip()
            if lbl in entities:
                entities[lbl].append(mention)
        outputs.append({"entities": entities})
        
        # Resolve provenance if files_content is provided
        if files_content:
            doc_id = resolve_provenance(text, files_content)
        else:
            doc_id = row.get('doc_id', "synthetic_generation")
        doc_ids.append(doc_id)
    
    df['output'] = outputs
    df['doc_id'] = doc_ids
        
    return df[['input', 'output', 'labels', 'doc_id']]

def main():
    logger.info("Initializing synthetic publication script...")
    env_vars = setup_local()

    repo_id = env_vars.get("HF_REPO_ID") or "your-username/archaeo-ner-greek"
    hf_token = env_vars.get("HF_TOKEN") or env_vars.get("HUGGING_FACE_HUB_TOKEN")
    hf_private = env_vars.get("HF_REPO_PRIVATE", "False").lower() == "true"

    if not repo_id:
        logger.error("HF_REPO_ID is missing in the environment.")
        sys.exit(1)
    if not hf_token:
        logger.error("HF_TOKEN is missing in the environment.")
        sys.exit(1)
        
    # 1. Fetch default train split from HF to curate against
    logger.info(f"Loading reference training split from HF repository: {repo_id}...")
    try:
        ds_gold = load_dataset(repo_id, name="default", token=hf_token)
        df_gold_train = ds_gold["train"].to_pandas()
        logger.info(f"Reference gold training split loaded ({len(df_gold_train)} samples).")
    except Exception as e:
        logger.error(f"Failed to load gold reference dataset: {e}")
        sys.exit(1)
        
    # 2. Load 1350 example synthetic JSON
    synth_path = Path(env_vars.get("SYNTHETIC_DATA_PATH", "data/synthetic_data_generation/synthetic_archaeology_real_seeded_gemini25flash_n1350.json"))
    if not synth_path.exists():
        logger.error(f"Synthetic JSON file not found at: {synth_path}")
        sys.exit(1)
        
    logger.info(f"Loading raw synthetic data from: {synth_path}...")
    df_raw_synthetic = pd.read_json(synth_path)
    logger.info(f"Loaded {len(df_raw_synthetic)} raw synthetic samples.")
    
    # 3. Generate curated synthetic data (currently used in training)
    logger.info("Curating active synthetic training subset (dynamic filtration + stratification)...")
    ratio = float(env_vars.get("GLINER_SYNTHETIC_RATIO", 2.0))
    df_curated_synthetic = curate_synthetic_data(df_raw_synthetic, df_gold_train, ratio=ratio, seed=42)
    logger.info(f"Curated active subset: {len(df_curated_synthetic)} samples.")
    
    # 4. Define labels master set
    all_labels = set()
    for labels_list in df_gold_train['labels']:
        for ent in labels_list:
            all_labels.add(ent['label'])
    all_labels = sorted(list(all_labels))
    logger.info(f"Master label set: {all_labels}")
    
    # Load raw text files to map provenance
    extra_texts_dir = Path("data/extra_texts")
    files_content = {}
    if extra_texts_dir.exists():
        logger.info(f"Loading files from {extra_texts_dir} to resolve provenance...")
        for filename in sorted(os.listdir(extra_texts_dir)):
            if filename.endswith(".txt"):
                filepath = extra_texts_dir / filename
                with open(filepath, "r", encoding="utf-8") as f:
                    files_content[filename] = f.read().strip()
        logger.info(f"Loaded {len(files_content)} files for provenance mapping.")
    else:
        logger.warning(f"Extra texts directory not found at: {extra_texts_dir}. Skipping provenance resolution.")
        
    # 5. Format datasets
    logger.info("Formatting datasets to GLiNER2 specifications and mapping provenance...")
    df_raw_fmt = prepare_gliner2_format(df_raw_synthetic, all_labels, files_content)
    df_curated_fmt = prepare_gliner2_format(df_curated_synthetic, all_labels, files_content)
    
    # 6. Define Features schema
    features = Features({
        'input': Value('string'),
        'output': {
            'entities': {lbl: Sequence(Value('string')) for lbl in all_labels}
        },
        'labels': [Features({
            'label': Value('string'),
            'start': Value('int64'),
            'end': Value('int64')
        })],
        'doc_id': Value('string')
    })
    
    # 7. Convert to HF Dataset Dict
    raw_ds = Dataset.from_pandas(df_raw_fmt, features=features, preserve_index=False)
    curated_ds = Dataset.from_pandas(df_curated_fmt, features=features, preserve_index=False)
    
    dataset_dict = DatasetDict({
        "raw_1350": raw_ds,
        "curated_520": curated_ds
    })
    
    # 8. Push to Hub under config "synthetic"
    logger.info(f"Pushing configuration 'synthetic' to Hugging Face Hub: {repo_id}...")
    try:
        dataset_dict.push_to_hub(
            repo_id=repo_id,
            config_name="synthetic",
            token=hf_token,
            private=hf_private
        )
        logger.info(f"✅ SUCCESS: Synthetic subset successfully pushed under configuration 'synthetic'.")
        logger.info(f"   - Split 'raw_1350': {len(raw_ds)} examples")
        logger.info(f"   - Split 'curated_520': {len(curated_ds)} examples")
    except Exception as e:
        logger.error(f"Failed to push to Hugging Face: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
