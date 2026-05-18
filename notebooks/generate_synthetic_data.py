# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # **API-based Few-Shot GLiNER Synthetic Data Generator**
#
# This script integrates:
# 1. **API-based Generation** using a customizable chat completions model.
# 2. **Few-Shot Grounding** using the official, clean, deduplicated Hugging Face dataset.
# 3. **GLiNER Training Readiness** using tokenization and exact token-span index alignment.
# 4. **Flat NER Support**: Greedily resolves nested/overlapping annotations for flat NER compatibility.
# 5. **Fail-Safe Checkpointing**: Flushes generated data to disk incrementally after every single batch.
#
# Outputs are directly compatible with GLiNER training requirements.

# %%
import os
import re
import json
import random
import logging
from collections import defaultdict
from typing import Optional
from openai import OpenAI

# Set up logging using the repository's standard configuration utility
try:
    from archaeo_ner_greek.logging_config import setup_logging
    setup_logging(log_file="data/synthetic_data_generation/synthetic_generation.log")
except ImportError:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- ENVIRONMENT HELPERS ---
def is_colab():
    """Detect if the script is running in Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False

if not is_colab():
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except ImportError:
        pass

def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Retrieve secret from Colab userdata or environment variables."""
    if is_colab():
        from google.colab import userdata
        try:
            return userdata.get(key)
        except Exception:
            return default
    return os.getenv(key, default)

# --- CONFIGURATION VARIABLES ---
# If a GOOGLE_API_KEY is found, default to gemini-2.5-flash; otherwise fall back to OpenAI (gpt-4o)
if get_secret("GOOGLE_API_KEY"):
    MODEL_NAME = get_secret("GEMINI_MODEL_NAME") or "gemini-2.5-flash"
else:
    MODEL_NAME = get_secret("OPENAI_MODEL_NAME") or "gpt-4o"
NUM_BATCHES = int(get_secret("SYNTHETIC_NUM_BATCHES", "3"))
NUM_SAMPLES_PER_BATCH = int(get_secret("SYNTHETIC_SAMPLES_PER_BATCH", "5"))
SEED = int(get_secret("SYNTHETIC_SEED", "42"))
FLATTEN_SPANS = get_secret("SYNTHETIC_FLATTEN_SPANS", "True").lower() in ("true", "1", "yes")
HF_REPO_ID_DEFAULT = get_secret("HF_REPO_ID", "Stalexan/archaeo-ner-greek")

# Set seed for reproducible sampling
random.seed(SEED)

# %% [markdown]
# ## 1. Greek Archaeology Guidelines

# %%
GUIDELINES = {
    "artefact": "movable archaeological find/inscription in Greek. e.g. αγγείο, νομίσματα, επιγραφή",
    "period": "historical era. e.g. ἑλληνιστικὴ περίοδος, κλασική εποχή",
    "location": "geographic names. e.g. Ρόδος, Αθήνα, Κνωσός",
    "context": "archaeological layer or structure. e.g. τάφοι, τείχη, στρώμα καταστροφής",
    "material": "object substance. e.g. χρυσός, πηλός, χαλκός",
    "person": "historical figures, deities, or annotator-named individuals. e.g. Απόλλων, Αλέξανδρος",
    "species": "biological entities (flora/fauna). e.g. δρυς, ελιά",
    "feature": "artistic style, architectural element, or motif."
}

# %% [markdown]
# ## 2. Tokenizer & GLiNER Span Alignment functions

# %%
def tokenize_text(text):
    """Tokenize the input text into a list of tokens, matching GLiNER logic."""
    return re.findall(r'\w+(?:[-_]\w+)*|\S', text)

def extract_entities(text, entities_list):
    """
    Converts raw text and a list of entity dicts into token-aligned span index tuples
    compatible with GLiNER: (start_token_idx, end_token_idx, label)
    """
    tokens = tokenize_text(text)
    spans = []
    
    for entity_item in entities_list:
        entity_name = str(entity_item.get("entity", "")).strip()
        entity_types = entity_item.get("types", [])
        if not entity_name or not entity_types:
            continue
            
        entity_tokens = tokenize_text(entity_name)
        n_tokens = len(entity_tokens)
        
        # Find all occurrences of the entity tokens in the main text tokens
        for i in range(len(tokens) - n_tokens + 1):
            if " ".join(tokens[i:i + n_tokens]).lower() == " ".join(entity_tokens).lower():
                for label in entity_types:
                    clean_label = label.lower().replace('_', ' ').strip()
                    spans.append([i, i + n_tokens - 1, clean_label])
                    
    return {"tokenized_text": tokens, "ner": spans}

# %% [markdown]
# ## 3. Few-Shot Pool Loader (Hugging Face with Offline Fallback)

# %%
class FewShotPoolLoader:
    """Loads few-shot examples from the Hugging Face dataset (Train split) or re-raises error."""
    
    STATIC_FALLBACK = [
        {
            "text": "Στο δυτικό τομέα του τάφου βρέθηκε ένα χάλκινο αγγείο της γεωμετρικής περιόδου.",
            "entities": [
                {"entity": "τάφου", "types": ["context"]},
                {"entity": "χάλκινο", "types": ["material"]},
                {"entity": "αγγείο", "types": ["artefact"]},
                {"entity": "γεωμετρικής περιόδου", "types": ["period"]}
            ]
        },
        {
            "text": "Κατά τις ανασκαφές στην αρχαία Κνωσό, οι αρχαιολόγοι εντόπισαν τμήμα του ανακτόρου.",
            "entities": [
                {"entity": "Κνωσό", "types": ["location"]},
                {"entity": "ανακτόρου", "types": ["context"]}
            ]
        },
        {
            "text": "Η μαρμάρινη κεφαλή του Απόλλωνα αποκαλύφθηκε κάτω από το στρώμα καταστροφής.",
            "entities": [
                {"entity": "μαρμάρινη", "types": ["material"]},
                {"entity": "κεφαλή", "types": ["artefact"]},
                {"entity": "Απόλλωνα", "types": ["person"]},
                {"entity": "στρώμα καταστροφής", "types": ["context"]}
            ]
        }
    ]

    @classmethod
    def load(cls, repo_id=HF_REPO_ID_DEFAULT, hf_token=None):
        try:
            from datasets import load_dataset
            token = hf_token or get_secret("HF_TOKEN") or get_secret("HUGGING_FACE_HUB_TOKEN")
            
            logger.info(f"Attempting to load train pool from Hugging Face: {repo_id}...")
            ds = load_dataset(repo_id, name="default", token=token)
            train_df = ds["train"].to_pandas()
            
            pool = []
            for _, row in train_df.iterrows():
                text = row.get("input")
                labels = row.get("labels")
                if not text or labels is None:
                    continue
                
                entities = []
                for ent in labels:
                    start = ent.get("start")
                    end = ent.get("end")
                    label = ent.get("label")
                    if start is not None and end is not None and label is not None:
                        ent_text = text[start:end]
                        entities.append({
                            "entity": ent_text,
                            "types": [label.lower().strip()]
                        })
                pool.append({
                    "text": text,
                    "entities": entities
                })
            
            if pool:
                logger.info(f"Successfully loaded {len(pool)} grounded records from Hugging Face dataset: {repo_id}")
                return pool
        except Exception as e:
            logger.error(f"FATAL: Hugging Face load failed for {repo_id}: {e}")
            raise e

# %% [markdown]
# ## 4. API-based Synthetic Generator

# %%
class OpenAISyntheticGenerator:
    def __init__(self, api_key, model_name=MODEL_NAME):
        base_url = get_secret("OPENAI_BASE_URL") or get_secret("LITELLM_BASE_URL") or get_secret("OPENAI_API_BASE")
        if base_url:
            logger.info(f"Using custom API base URL: {base_url}")
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=30.0)
        self.model = model_name
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def generate_batch(self, pool, num_samples_per_batch=NUM_SAMPLES_PER_BATCH):
        # Sample few-shot examples
        samples = random.sample(pool, k=min(3, len(pool)))
        examples_str = ""
        for idx, s in enumerate(samples):
            examples_str += f"### Example {idx + 1}:\n"
            examples_str += json.dumps(s, ensure_ascii=False, indent=2) + "\n\n"

        prompt = f"""You are a professional Greek Archaeologist and Domain Expert.
Your objective is to produce realistic Greek archaeological text passages that contain named entities, properly annotated according to our taxonomic guidelines.

Guidelines:
{json.dumps(GUIDELINES, ensure_ascii=False, indent=2)}

Format Requirements:
- Output MUST be structured in JSON containing a "samples" list.
- Each sample must contain a "text" key (fluent Greek sentence) and an "entities" list.
- Each entity must map to its exact string "entity" and its matching classification tag in "types" (lowercase).

{examples_str}

Task:
Generate {num_samples_per_batch} highly realistic, varied archaeological texts with entity annotations in the exact JSON format shown in the examples.
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a specialized system that generates Greek archaeological NER datasets in strict JSON format."},
                    {"role": "user", "content": prompt}
                ],
                response_format={ "type": "json_object" }
            )
            usage = response.usage
            if usage:
                self.total_prompt_tokens += usage.prompt_tokens
                self.total_completion_tokens += usage.completion_tokens
                logger.info(f"[{self.model}] Batch Tokens: Input={usage.prompt_tokens}, Output={usage.completion_tokens}")
            raw_content = json.loads(response.choices[0].message.content)
            return raw_content.get("samples", [])
        except Exception as e:
            logger.error(f"Error during API generation batch using {self.model}: {e}")
            return []

# %%
class GeminiSyntheticGenerator:
    def __init__(self, api_key, model_name=MODEL_NAME):
        self.model = model_name
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            timeout=30.0
        )
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def generate_batch(self, pool, num_samples_per_batch=NUM_SAMPLES_PER_BATCH):
        # Sample few-shot examples
        samples = random.sample(pool, k=min(3, len(pool)))
        examples_str = ""
        for idx, s in enumerate(samples):
            examples_str += f"### Example {idx + 1}:\n"
            examples_str += json.dumps(s, ensure_ascii=False, indent=2) + "\n\n"

        prompt = f"""You are a professional Greek Archaeologist and Domain Expert.
Your objective is to produce realistic Greek archaeological text passages that contain named entities, properly annotated according to our taxonomic guidelines.

Guidelines:
{json.dumps(GUIDELINES, ensure_ascii=False, indent=2)}

Format Requirements:
- Output MUST be structured in JSON containing a "samples" list.
- Each sample must contain a "text" key (fluent Greek sentence) and an "entities" list.
- Each entity must map to its exact string "entity" and its matching classification tag in "types" (lowercase).

{examples_str}

Task:
Generate {num_samples_per_batch} highly realistic, varied archaeological texts with entity annotations in the exact JSON format shown in the examples.
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a specialized system that generates Greek archaeological NER datasets in strict JSON format."},
                    {"role": "user", "content": prompt}
                ],
                response_format={ "type": "json_object" }
            )
            usage = response.usage
            if usage:
                self.total_prompt_tokens += usage.prompt_tokens
                self.total_completion_tokens += usage.completion_tokens
                logger.info(f"[{self.model}] Batch Tokens: Input={usage.prompt_tokens}, Output={usage.completion_tokens}")
            raw_content = json.loads(response.choices[0].message.content)
            return raw_content.get("samples", [])
        except Exception as e:
            logger.error(f"Error during Gemini API generation batch using {self.model}: {e}")
            return []

# %% [markdown]
# ## 5. Execution Pipeline

def get_synthetic_generator(model_name: str):
    """Factory to resolve API keys, handle fallbacks, and instantiate the generator."""
    google_key = get_secret("GOOGLE_API_KEY")
    openai_key = get_secret("OPENAI_API_KEY")

    if not google_key and not openai_key:
        raise ValueError("CRITICAL: Both GOOGLE_API_KEY and OPENAI_API_KEY are missing from environment.")

    # Dynamic provider routing with automatic key fallbacks
    is_gemini = str(model_name).lower().startswith("gemini")

    if is_gemini and not google_key:
        logger.warning(f"Gemini model '{model_name}' was selected, but GOOGLE_API_KEY is undefined. Falling back to OpenAI (gpt-4o)...")
        model_name = "gpt-4o"
        is_gemini = False

    if not is_gemini and not openai_key:
        logger.warning(f"OpenAI model '{model_name}' was selected, but OPENAI_API_KEY is undefined. Falling back to Gemini (gemini-2.5-flash)...")
        model_name = "gemini-2.5-flash"
        is_gemini = True

    # Instantiate generator
    if is_gemini:
        logger.info(f"Initializing Gemini generator using model '{model_name}'...")
        return GeminiSyntheticGenerator(api_key=google_key, model_name=model_name)
    else:
        logger.info(f"Initializing OpenAI/OpenRouter generator using model '{model_name}'...")
        return OpenAISyntheticGenerator(api_key=openai_key, model_name=model_name)

# %%
def flatten_spans(labels):
    """Greedily resolves overlapping/nested spans by keeping the longest span first."""
    sorted_labels = sorted(labels, key=lambda x: (x["end"] - x["start"]), reverse=True)
    kept_labels = []
    for lbl in sorted_labels:
        s1, e1 = lbl["start"], lbl["end"]
        overlap = False
        for kept in kept_labels:
            s2, e2 = kept["start"], kept["end"]
            if s1 < e2 and s2 < e1:
                overlap = True
                break
        if not overlap:
            kept_labels.append(lbl)
    return sorted(kept_labels, key=lambda x: x["start"])

# %%
def convert_to_char_spans(text, entities_list):
    """Converts LLM entities representation into character-level spans matching Hugging Face schema."""
    labels = []
    aligned_count = 0
    skipped_count = 0
    for entity_item in entities_list:
        entity_name = str(entity_item.get("entity", "")).strip()
        entity_types = entity_item.get("types", [])
        if not entity_name or not entity_types:
            skipped_count += 1
            continue
        
        # Find all character occurrences of the entity in the raw text
        start_idx = 0
        found = False
        while True:
            start_idx = text.lower().find(entity_name.lower(), start_idx)
            if start_idx == -1:
                break
            found = True
            end_idx = start_idx + len(entity_name)
            # Add for each label type, normalized to uppercase
            for label in entity_types:
                clean_label = label.upper().replace('_', ' ').strip()
                labels.append({
                    "label": clean_label,
                    "start": start_idx,
                    "end": end_idx
                })
            start_idx += 1
        
        if found:
            aligned_count += 1
        else:
            skipped_count += 1
            
    # Flatten spans if configuration is active
    flattened_count = 0
    if FLATTEN_SPANS:
        orig_len = len(labels)
        labels = flatten_spans(labels)
        flattened_count = orig_len - len(labels)

    return {"input": text, "labels": labels}, aligned_count, skipped_count, flattened_count

# %%
def run_generation_pipeline(num_batches=NUM_BATCHES, num_samples_per_batch=NUM_SAMPLES_PER_BATCH, model_name=None):
    if model_name is None:
        model_name = MODEL_NAME

    # Resolve the generator modularly via the factory
    try:
        generator = get_synthetic_generator(model_name)
    except ValueError as e:
        logger.error(e)
        return

    # Load source pool from HF
    repo_id = HF_REPO_ID_DEFAULT
    pool = FewShotPoolLoader.load(repo_id=repo_id)
    
    # Establish target output file before starting loop
    output_dir = "data/synthetic_data_generation"
    os.makedirs(output_dir, exist_ok=True)
    normalized_model = str(model_name).lower().replace("-", "").replace(".", "")
    total_target_samples = num_batches * num_samples_per_batch
    output_filename = f"synthetic_archaeology_{normalized_model}_n{total_target_samples}.json"
    output_file = os.path.join(output_dir, output_filename)
    
    # Truncate/remove old output file if starting a fresh run
    if os.path.exists(output_file):
        try:
            os.remove(output_file)
        except Exception:
            pass

    logger.info(f"Starting generation of {total_target_samples} samples using model {model_name}...")
    if FLATTEN_SPANS:
        logger.info("Greedy span flattening is enabled. Overlapping/nested spans will be resolved.")
    logger.info(f"Target Output File: {output_file} (Flushed incrementally after every batch)")

    total_aligned = 0
    total_skipped = 0
    total_flattened = 0
    label_distribution = defaultdict(int)
    
    hf_dataset = []

    for i in range(num_batches):
        logger.info(f"Generating batch {i+1}/{num_batches}...")
        batch = generator.generate_batch(pool, num_samples_per_batch=num_samples_per_batch)
        
        # Align this batch immediately
        batch_hf = []
        for s in batch:
            text = s.get("text")
            entities = s.get("entities")
            if text and entities:
                aligned, aligned_cnt, skipped_cnt, flattened_cnt = convert_to_char_spans(text, entities)
                batch_hf.append(aligned)
                total_aligned += aligned_cnt
                total_skipped += skipped_cnt
                total_flattened += flattened_cnt
                for lbl in aligned["labels"]:
                    label_distribution[lbl["label"]] += 1
        
        # Append batch to running in-memory list and instantly flush/save to disk
        hf_dataset.extend(batch_hf)
        total_processed_records = len(hf_dataset)
        
        with open(output_file, "w", encoding="utf-8") as f_out:
            json.dump(hf_dataset, f_out, ensure_ascii=False, indent=2)
            
        logger.info(f"Batch {i+1}/{num_batches} successfully processed and flushed to disk. (Cumulative: {total_processed_records} samples)")
        
    logger.info(f"Hugging Face-ready synthetic dataset successfully saved to: {output_file}")
    logger.info(f"Total processed training records: {total_processed_records}")

    # Print Diagnostics & Insights
    logger.info("=== DATASET DIAGNOSTICS & QUALITY ASSURANCE ===")
    logger.info(f"Successfully Aligned Entities:      {total_aligned}")
    logger.info(f"Skipped / Hallucinated Entities:    {total_skipped}")
    if FLATTEN_SPANS:
        logger.info(f"Resolved / Flattened Nested Spans:  {total_flattened}")
    if total_aligned + total_skipped > 0:
        alignment_rate = (total_aligned / (total_aligned + total_skipped)) * 100
        logger.info(f"Entity Alignment Success Rate:      {alignment_rate:.1f}%")
    logger.info("Entity Type Distribution:")
    for lbl, count in sorted(label_distribution.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  - {lbl}: {count}")

    if hasattr(generator, "total_prompt_tokens") and generator.total_prompt_tokens > 0:
        logger.info("=== CUMULATIVE TOKEN & CREDIT CONSUMPTION ===")
        logger.info(f"Total Input (Prompt) Tokens:      {generator.total_prompt_tokens}")
        logger.info(f"Total Output (Completion) Tokens:  {generator.total_completion_tokens}")
        
        if "gemini" in str(model_name).lower():
            # Standard Gemini 2.5 Flash pricing: Input = $0.075 / 1M, Output = $0.30 / 1M
            cost = (generator.total_prompt_tokens * 0.000000075) + (generator.total_completion_tokens * 0.00000030)
            logger.info(f"Estimated Cost (Google Pay-As-You-Go): ${cost:.6f} USD (Free Tier: $0.00)")
        elif "gpt-4o" in str(model_name).lower():
            # Standard GPT-4o pricing: Input = $2.50 / 1M, Output = $10.00 / 1M
            cost = (generator.total_prompt_tokens * 0.0000025) + (generator.total_completion_tokens * 0.000010)
            logger.info(f"Estimated Cost (OpenAI standard):      ${cost:.6f} USD")

# %%
if __name__ == "__main__":
    # Execute generation if run as script
    run_generation_pipeline(num_batches=NUM_BATCHES, num_samples_per_batch=NUM_SAMPLES_PER_BATCH)
