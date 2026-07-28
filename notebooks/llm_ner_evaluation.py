# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # LLM Baselines for Archaeological NER
#
# Few-shot evaluation of generative LLMs on the Archaeo-NER-Greek test set,
# for comparison against GLiNER2 (span-based, fine-tuned).
#
# Models:
# - **Krikri 8B** (`ilsp/Llama-Krikri-8B-Instruct-v1.5`): Greek-specific, local inference
# - **Gemma 4 27B** (via OpenRouter API): multilingual, large-scale

# %% [markdown]
# ## Environment Setup

# %%
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from dotenv import dotenv_values, find_dotenv, load_dotenv

from archaeo_ner_greek.logging_config import setup_logging

# Load environment
env_path = find_dotenv()
if env_path:
    load_dotenv(env_path, override=True)
    env_vars = dotenv_values(env_path)
else:
    env_vars = {}

log_file = setup_logging()
logger = logging.getLogger(__name__)
logger.info(f">>> Logging to: {log_file}")

BASE_DIR = Path(os.getcwd())
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = DATA_DIR / "llm_baselines"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load Dataset & Entity Schema

# %%
import archaeo_ner_greek

# Load HF dataset (same splits as GLiNER2)
repo_id = env_vars.get("HF_REPO_ID", "Stalexan/archaeo-ner-greek")
hf_token = env_vars.get("HF_TOKEN")
ds = load_dataset(repo_id, name="default", token=hf_token, revision="dev")

df_train = ds["train"].to_pandas()
df_val = ds["validation"].to_pandas()
df_test = ds["test"].to_pandas()

logger.info(f"Splits: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")

# Load entity definitions
PACKAGE_ROOT = Path(archaeo_ner_greek.__file__).parent
GUIDELINES_PATH = PACKAGE_ROOT / "resources" / "archaeoner_labels_definitions_v12_st.json"
with open(GUIDELINES_PATH, "r", encoding="utf-8") as f:
    entity_descriptions = json.load(f)

LABELS = list(entity_descriptions.keys())
logger.info(f"Labels: {LABELS}")


# %% [markdown]
# ## Few-Shot Example Selection

# %%
def select_few_shot_examples(df, k=5, seed=42):
    """Select k diverse training examples for few-shot prompting.

    Strategy: sample examples that collectively cover all entity categories.
    """
    import random
    random.seed(seed)

    # Parse entities column (stored as JSON string or dict)
    examples = []
    for _, row in df.iterrows():
        entities = row["entities"] if isinstance(row["entities"], dict) else json.loads(row["entities"])
        # Count non-empty categories
        active_labels = {lbl for lbl, spans in entities.items() if spans}
        examples.append({"text": row["text"], "entities": entities, "active_labels": active_labels})

    # Greedy coverage: pick examples that maximize label coverage
    selected = []
    covered = set()
    remaining = examples.copy()

    while len(selected) < k and remaining:
        # Score by number of new labels covered
        best = max(remaining, key=lambda ex: len(ex["active_labels"] - covered))
        selected.append(best)
        covered |= best["active_labels"]
        remaining.remove(best)

    return selected


few_shot_examples = select_few_shot_examples(df_train, k=5)
logger.info(f"Selected {len(few_shot_examples)} few-shot examples covering labels: "
            f"{set().union(*(ex['active_labels'] for ex in few_shot_examples))}")


# %% [markdown]
# ## Prompt Construction

# %%
def build_system_prompt(entity_descriptions):
    """Build the system prompt with entity definitions."""
    label_defs = "\n".join(
        f"- **{label}**: {desc}" for label, desc in entity_descriptions.items()
    )
    return f"""You are an expert Named Entity Recognition (NER) system for Greek archaeological texts.

Given a text, extract all named entities and classify each into exactly one of the following categories:

{label_defs}

Output format: Return a JSON object where keys are entity category names and values are lists of extracted entity text spans (exact substrings from the input). Include all categories, using empty lists for categories with no entities found.

Rules:
- Extract exact text spans as they appear in the input (no paraphrasing).
- Each entity must belong to exactly one category.
- If no entities are found for a category, return an empty list.
- Return ONLY the JSON object, no other text."""


def build_few_shot_messages(system_prompt, few_shot_examples, query_text):
    """Build the full message sequence for few-shot prompting."""
    messages = [{"role": "system", "content": system_prompt}]

    for ex in few_shot_examples:
        messages.append({"role": "user", "content": ex["text"]})
        # Build the expected output (only non-empty + empty categories)
        output = {label: ex["entities"].get(label, []) for label in LABELS}
        messages.append({"role": "assistant", "content": json.dumps(output, ensure_ascii=False)})

    messages.append({"role": "user", "content": query_text})
    return messages


SYSTEM_PROMPT = build_system_prompt(entity_descriptions)


# %% [markdown]
# ## Response Parsing

# %%
def parse_llm_response(response_text):
    """Parse LLM response into entity dict. Handles malformed JSON gracefully."""
    # Strip markdown code fences if present
    text = response_text.strip()
    if text.startswith("```"):
        # Remove ```json or ``` prefix and ``` suffix
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse LLM response: {text[:200]}...")
        return {label: [] for label in LABELS}

    # Normalize: ensure all labels present, values are lists of strings
    result = {}
    for label in LABELS:
        val = parsed.get(label, [])
        if isinstance(val, list):
            result[label] = [str(v) for v in val]
        else:
            result[label] = []

    return result


# %% [markdown]
# ## Model Clients

# %%
# --- Krikri 8B (Local via transformers) ---

def load_krikri_model():
    """Load Krikri 8B for local inference."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "ilsp/Llama-Krikri-8B-Instruct-v1.5"
    logger.info(f"Loading {model_id}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    logger.info(f"Krikri loaded on {model.device}")
    return model, tokenizer


def predict_krikri(model, tokenizer, messages, max_new_tokens=2048):
    """Run inference with Krikri 8B."""
    input_ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to(model.device)

    with __import__("torch").no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy for reproducibility
            temperature=None,
            top_p=None,
        )

    # Decode only generated tokens
    generated = output_ids[0][input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


# --- Gemma 4 27B (OpenRouter API) ---

def predict_openrouter(messages, model_id="google/gemma-4-27b-it", max_tokens=2048):
    """Run inference via OpenRouter API."""
    import requests

    api_key = env_vars.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY"))
    base_url = env_vars.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

    response = requests.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,  # greedy for reproducibility
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# %% [markdown]
# ## Evaluation Metrics

# %%
def compute_ner_metrics(predictions, ground_truths, labels):
    """Compute micro P/R/F1 and per-label metrics.

    Args:
        predictions: list of dicts {label: [span, ...]}
        ground_truths: list of dicts {label: [span, ...]}
        labels: list of label names
    """
    tp, fp, fn = 0, 0, 0
    label_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for pred, gt in zip(predictions, ground_truths):
        pred_spans = [(t, lbl) for lbl in labels for t in pred.get(lbl, [])]
        gt_spans = [(t, lbl) for lbl in labels for t in gt.get(lbl, [])]

        temp_gt = gt_spans.copy()
        for p in pred_spans:
            if p in temp_gt:
                tp += 1
                label_stats[p[1]]["tp"] += 1
                temp_gt.remove(p)
            else:
                fp += 1
                label_stats[p[1]]["fp"] += 1

        for g in temp_gt:
            fn += 1
            label_stats[g[1]]["fn"] += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    per_label = {}
    for lbl in labels:
        s = label_stats[lbl]
        l_p = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) > 0 else 0
        l_r = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) > 0 else 0
        l_f1 = 2 * l_p * l_r / (l_p + l_r) if (l_p + l_r) > 0 else 0
        per_label[lbl] = {"precision": l_p, "recall": l_r, "f1": l_f1,
                          "tp": s["tp"], "fp": s["fp"], "fn": s["fn"]}

    return {
        "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
        "per_label": per_label,
    }


# %% [markdown]
# ## Run Evaluation

# %%
def evaluate_model(predict_fn, df_test, few_shot_examples, model_name):
    """Run few-shot NER evaluation on the test set.

    Args:
        predict_fn: callable(messages) -> str (raw LLM response)
        df_test: test DataFrame
        few_shot_examples: list of few-shot example dicts
        model_name: identifier for logging/saving
    """
    predictions = []
    ground_truths = []
    raw_responses = []

    logger.info(f"Evaluating {model_name} on {len(df_test)} test samples...")

    for i, (_, row) in enumerate(df_test.iterrows()):
        text = row["text"]
        gt = row["entities"] if isinstance(row["entities"], dict) else json.loads(row["entities"])

        messages = build_few_shot_messages(SYSTEM_PROMPT, few_shot_examples, text)

        try:
            response = predict_fn(messages)
            parsed = parse_llm_response(response)
        except Exception as e:
            logger.error(f"[{model_name}] Sample {i} failed: {e}")
            response = ""
            parsed = {label: [] for label in LABELS}

        predictions.append(parsed)
        ground_truths.append(gt)
        raw_responses.append(response)

        if (i + 1) % 20 == 0:
            logger.info(f"[{model_name}] Processed {i + 1}/{len(df_test)}")

    # Compute metrics
    metrics = compute_ner_metrics(predictions, ground_truths, LABELS)
    logger.info(f"[{model_name}] Micro-F1: {metrics['f1']:.4f} | "
                f"P: {metrics['precision']:.4f} | R: {metrics['recall']:.4f}")

    # Save raw results
    results_path = RESULTS_DIR / f"{model_name}_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": model_name,
            "metrics": metrics,
            "predictions": predictions,
            "raw_responses": raw_responses,
        }, f, ensure_ascii=False, indent=2)
    logger.info(f"[{model_name}] Results saved to {results_path}")

    return metrics


# %% [markdown]
# ## Evaluate Krikri 8B (Local)

# %%
# krikri_model, krikri_tokenizer = load_krikri_model()
# krikri_metrics = evaluate_model(
#     predict_fn=lambda msgs: predict_krikri(krikri_model, krikri_tokenizer, msgs),
#     df_test=df_test,
#     few_shot_examples=few_shot_examples,
#     model_name="krikri_8b",
# )

# %% [markdown]
# ## Evaluate Gemma 4 27B (OpenRouter)

# %%
# gemma_metrics = evaluate_model(
#     predict_fn=lambda msgs: predict_openrouter(msgs, model_id="google/gemma-4-27b-it"),
#     df_test=df_test,
#     few_shot_examples=few_shot_examples,
#     model_name="gemma4_27b",
# )

# %% [markdown]
# ## Results Summary

# %%
def print_results_table(results_dict):
    """Print a comparison table of all model results."""
    from tabulate import tabulate

    # Summary row
    rows = []
    for name, metrics in results_dict.items():
        rows.append([name, f"{metrics['precision']:.4f}", f"{metrics['recall']:.4f}",
                      f"{metrics['f1']:.4f}"])

    print("\n" + tabulate(rows, headers=["Model", "Precision", "Recall", "F1"], tablefmt="grid"))

    # Per-label breakdown
    for name, metrics in results_dict.items():
        print(f"\n--- {name} Per-Label ---")
        label_rows = []
        for lbl in LABELS:
            s = metrics["per_label"].get(lbl, {"precision": 0, "recall": 0, "f1": 0})
            label_rows.append([lbl, f"{s['precision']:.4f}", f"{s['recall']:.4f}", f"{s['f1']:.4f}"])
        print(tabulate(label_rows, headers=["Label", "P", "R", "F1"], tablefmt="grid"))


# Uncomment after running evaluations:
# print_results_table({"Krikri 8B": krikri_metrics, "Gemma 4 27B": gemma_metrics})
