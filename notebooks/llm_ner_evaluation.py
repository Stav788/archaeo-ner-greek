import argparse
import csv
import importlib.metadata
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset
from dotenv import dotenv_values, find_dotenv, load_dotenv

import archaeo_ner_greek
from archaeo_ner_greek.logging_config import setup_logging
from datetime import datetime


ENV_PATH = find_dotenv()
if ENV_PATH:
    load_dotenv(ENV_PATH, override=True)
    ENV_VARS = dotenv_values(ENV_PATH)
else:
    ENV_VARS = {}

BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"

# Base directory for logs
RESULTS_DIR = BASE_DIR / "logs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Set up initial logging
LOG_FILE = setup_logging(log_file=str(RESULTS_DIR / "execution.log"))
LOGGER = logging.getLogger(__name__)
LOGGER.info("Evaluation log initialized: %s", LOG_FILE)


REPO_ID = ENV_VARS.get("HF_REPO_ID") or "your-username/archaeo-ner-greek"
HF_TOKEN = ENV_VARS.get("HF_TOKEN")
DATASET_REVISION = "dev"

MODEL_COSTS = {
    "google/gemma-4-26b-a4b-it": {"input_per_1m_tokens": 0.07, "output_per_1m_tokens": 0.34},
    "meta-llama/llama-3.1-8b-instruct": {"input_per_1m_tokens": 0.05, "output_per_1m_tokens": 0.08},
    "qwen/qwen3-32b": {"input_per_1m_tokens": 0.08, "output_per_1m_tokens": 0.28},
    "ilsp/Llama-Krikri-8B-Instruct-v1.5": {"input_per_1m_tokens": 0.0, "output_per_1m_tokens": 0.0},
}

MODEL_PARAMETER_COUNTS = {
    "google/gemma-4-26b-a4b-it": 26_000_000_000,
    "meta-llama/llama-3.1-8b-instruct": 8_000_000_000,
    "qwen/qwen3-32b": 32_000_000_000,
    "ilsp/Llama-Krikri-8B-Instruct-v1.5": 8_200_000_000,
}

PACKAGE_ROOT = Path(archaeo_ner_greek.__file__).parent
GUIDELINES_PATH = PACKAGE_ROOT / "resources" / "archaeoner_labels_definitions_v12_st.json"


def load_entity_schema():
    with open(GUIDELINES_PATH, "r", encoding="utf-8") as handle:
        entity_descriptions = json.load(handle)
    labels = list(entity_descriptions.keys())
    LOGGER.info("Loaded entity schema with %d labels: %s", len(labels), labels)
    return entity_descriptions, labels


ENTITY_DESCRIPTIONS, LABELS = load_entity_schema()


def _extract_row_text(row):
    """Extract sentence text from a dataset row.

    Supports the HF schema ('input') and legacy Argilla schema ('sentence_field').
    """
    text = row.get("input") or row.get("text") or row.get("sentence_field")
    if text is None:
        raise KeyError("Row must contain 'input', 'text', or 'sentence_field'.")
    return text


def _extract_row_entities(row):
    """Extract the entity dict ({label: [spans]}) from a dataset row.

    Extracts spans directly from the 'labels' column via character offsets,
    matching the logic used in GLiNER training to avoid numpy array truth-value errors.
    """
    text = _extract_row_text(row)
    entities = {lbl: [] for lbl in LABELS}
    labels = row.get("labels", [])
    
    # Handling both list of dicts (pandas) and HF dataset dict-of-lists format
    if isinstance(labels, list) or hasattr(labels, "__iter__"):
        for label_obj in labels:
            if isinstance(label_obj, dict):
                lbl = label_obj.get("label")
                start = label_obj.get("start")
                end = label_obj.get("end")
            else:
                lbl = label_obj["label"]
                start = label_obj["start"]
                end = label_obj["end"]
            
            if lbl and start is not None and end is not None:
                mention = text[start:end].strip()
                if lbl in entities:
                    entities[lbl].append(mention)
    return entities


def collect_environment_metadata():
    """Collect Python, library, and GPU version information for reproducibility."""
    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
    }
    for package_name in ("torch", "transformers", "datasets"):
        try:
            metadata[f"{package_name}_version"] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            metadata[f"{package_name}_version"] = None
    try:
        import torch
        if torch.cuda.is_available():
            metadata["gpu_name"] = torch.cuda.get_device_name(0)
            vram_bytes = torch.cuda.get_device_properties(0).total_mem
            metadata["gpu_vram_gb"] = round(vram_bytes / (1024 ** 3), 2)
            metadata["gpu_driver_version"] = torch.version.cuda
        else:
            metadata["gpu_name"] = None
            metadata["gpu_vram_gb"] = None
            metadata["gpu_driver_version"] = None
    except Exception:
        metadata["gpu_name"] = None
        metadata["gpu_vram_gb"] = None
        metadata["gpu_driver_version"] = None
    return metadata


def collect_run_configuration(model_id, k, seed, max_new_tokens, temperature,
                              do_sample, system_prompt):
    """Record model identity, hyperparameters, and prompt for reproducibility."""
    return {
        "model_id": model_id,
        "parameter_count": MODEL_PARAMETER_COUNTS.get(model_id),
        "few_shot_k": k,
        "few_shot_seed": seed,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": do_sample,
        "system_prompt": system_prompt,
    }


def generate_qualitative_report(test_df, predictions, labels, output_path):
    """Write a per-sentence, per-label TSV comparing gold and predicted spans.

    The output is intended for manual review by domain experts who may not
    use programming tools. Only rows with at least one gold or predicted
    span are included.
    """
    fieldnames = [
        "sample_index", "text", "label",
        "gold_spans", "predicted_spans",
        "false_positives", "false_negatives", "match",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row_idx, (_, row) in enumerate(test_df.iterrows()):
            gold = _extract_row_entities(row)
            pred = predictions[row_idx]
            for label in labels:
                gold_spans = set(gold.get(label, []))
                pred_spans = set(pred.get(label, []))
                if not gold_spans and not pred_spans:
                    continue
                fp = pred_spans - gold_spans
                fn = gold_spans - pred_spans
                if not gold_spans and not pred_spans:
                    match_status = "empty"
                elif gold_spans == pred_spans:
                    match_status = "exact"
                elif gold_spans & pred_spans:
                    match_status = "partial"
                else:
                    match_status = "mismatch"
                writer.writerow({
                    "sample_index": row_idx,
                    "text": _extract_row_text(row),
                    "label": label,
                    "gold_spans": "; ".join(sorted(gold_spans)),
                    "predicted_spans": "; ".join(sorted(pred_spans)),
                    "false_positives": "; ".join(sorted(fp)),
                    "false_negatives": "; ".join(sorted(fn)),
                    "match": match_status,
                })
    LOGGER.info("Qualitative report written to %s", output_path)


def select_few_shot_examples(df, k=5, seed=42):
    import random

    random.seed(seed)
    examples = []
    for _, row in df.iterrows():
        entities = _extract_row_entities(row)
        active_labels = {label for label, spans in entities.items() if spans}
        examples.append({"text": _extract_row_text(row), "entities": entities, "active_labels": active_labels})

    selected = []
    covered = set()
    remaining = examples.copy()
    while len(selected) < k and remaining:
        candidate = max(remaining, key=lambda ex: len(ex["active_labels"] - covered))
        selected.append(candidate)
        covered |= candidate["active_labels"]
        remaining.remove(candidate)

    LOGGER.info("Selected %d few-shot examples covering %d labels.", len(selected), len(covered))
    return selected


def build_system_prompt(entity_descriptions):
    label_defs = "\n".join(f"- **{label}**: {desc}" for label, desc in entity_descriptions.items())
    return f"""You are an expert Named Entity Recognition (NER) system for Greek archaeological texts.

Given a text, extract all named entities and classify each into exactly one of the following categories:

{label_defs}

Return a JSON object with keys equal to the entity category names and values equal to lists of exact text spans extracted from the input.

Rules:
- Use exact substrings from the source text.
- Do not paraphrase or normalize entity text.
- Each extracted span belongs to exactly one category.
- Return empty lists for categories with no matches.
- Return JSON only; no explanatory text."""


def build_few_shot_messages(system_prompt, few_shot_examples, query_text):
    messages = [{"role": "system", "content": system_prompt}]
    for example in few_shot_examples:
        output = {label: example["entities"].get(label, []) for label in LABELS}
        messages.append({"role": "user", "content": example["text"]})
        messages.append({"role": "assistant", "content": json.dumps(output, ensure_ascii=False)})
    messages.append({"role": "user", "content": query_text})
    return messages


SYSTEM_PROMPT = build_system_prompt(ENTITY_DESCRIPTIONS)


def _clean_code_block(text):
    value = text.strip()
    if value.startswith("```"):
        lines = [line for line in value.splitlines() if not line.strip().startswith("```")]
        value = "\n".join(lines).strip()
    return value


def parse_llm_response(response_text):
    raw_text = _clean_code_block(response_text or "")
    if not raw_text:
        return {label: [] for label in LABELS}

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        LOGGER.warning("Failed to parse LLM JSON response: %s", raw_text[:200])
        return {label: [] for label in LABELS}

    if not isinstance(parsed, dict):
        LOGGER.warning("LLM response did not contain a JSON object: %s", raw_text[:200])
        return {label: [] for label in LABELS}

    result = {}
    for label in LABELS:
        values = parsed.get(label, [])
        if not isinstance(values, list):
            result[label] = []
        else:
            result[label] = [str(value) for value in values]
    return result


def load_dataset_from_hf():
    ds = load_dataset(REPO_ID, name="default", token=HF_TOKEN, revision=DATASET_REVISION)
    train_df = ds["train"].to_pandas()
    val_df = ds["validation"].to_pandas()
    test_df = ds["test"].to_pandas()
    LOGGER.info("Dataset loaded: train=%d, validation=%d, test=%d", len(train_df), len(val_df), len(test_df))
    return train_df, val_df, test_df


def estimate_api_cost(model_id, usage):
    cost_table = MODEL_COSTS.get(model_id)
    if not cost_table:
        return 0.0

    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    input_cost = (prompt_tokens / 1_000_000.0) * cost_table["input_per_1m_tokens"]
    output_cost = (completion_tokens / 1_000_000.0) * cost_table["output_per_1m_tokens"]
    return input_cost + output_cost


def compute_ner_metrics(predictions, ground_truths, labels):
    tp = 0
    fp = 0
    fn = 0
    label_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for pred, gt in zip(predictions, ground_truths):
        pred_spans = [(span, label) for label in labels for span in pred.get(label, [])]
        gt_spans = [(span, label) for label in labels for span in gt.get(label, [])]

        remaining_gt = gt_spans.copy()
        for item in pred_spans:
            if item in remaining_gt:
                tp += 1
                label_stats[item[1]]["tp"] += 1
                remaining_gt.remove(item)
            else:
                fp += 1
                label_stats[item[1]]["fp"] += 1

        for item in remaining_gt:
            fn += 1
            label_stats[item[1]]["fn"] += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    per_label = {}
    for label in labels:
        stats = label_stats[label]
        label_precision = stats["tp"] / (stats["tp"] + stats["fp"]) if (stats["tp"] + stats["fp"]) > 0 else 0.0
        label_recall = stats["tp"] / (stats["tp"] + stats["fn"]) if (stats["tp"] + stats["fn"]) > 0 else 0.0
        label_f1 = (2 * label_precision * label_recall / (label_precision + label_recall)) if (label_precision + label_recall) > 0 else 0.0
        per_label[label] = {
            "precision": label_precision,
            "recall": label_recall,
            "f1": label_f1,
            "tp": stats["tp"],
            "fp": stats["fp"],
            "fn": stats["fn"],
        }

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "per_label": per_label,
    }


def load_krikri_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "ilsp/Llama-Krikri-8B-Instruct-v1.5"
    LOGGER.info("Loading local model: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    LOGGER.info("Model loaded on device: %s", getattr(model, "device", "unknown"))
    return model, tokenizer


def predict_krikri(model, tokenizer, messages, max_new_tokens=2048, constrained=False):
    import torch

    if constrained:
        import outlines
        from pydantic import BaseModel, Field

        class NEROutput(BaseModel):
            ARTEFACT: list[str] = Field(default_factory=list)
            PERIOD: list[str] = Field(default_factory=list)
            LOCATION: list[str] = Field(default_factory=list)
            CONTEXT: list[str] = Field(default_factory=list)
            MATERIAL: list[str] = Field(default_factory=list)
            SPECIES: list[str] = Field(default_factory=list)
            PERSON: list[str] = Field(default_factory=list)
            FEATURE: list[str] = Field(default_factory=list)

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        wrapped_model = outlines.models.from_transformers(model, tokenizer)
        
        # Outlines 1.3.x natively handles structured generation via the __call__ method
        res = wrapped_model(prompt, NEROutput, max_new_tokens=max_new_tokens)
        
        # Ensure it's a JSON string so `parse_llm_response` can handle it natively
        if hasattr(res, "model_dump_json"):
            decoded_text = res.model_dump_json()
        else:
            decoded_text = str(res)
        
        # Estimate usage since outlines wraps generation
        input_len = len(tokenizer.encode(prompt))
        output_len = len(tokenizer.encode(decoded_text))
        usage = {
            "prompt_tokens": input_len,
            "completion_tokens": output_len,
            "total_tokens": input_len + output_len,
        }
        return {"text": decoded_text, "usage": usage}

    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Calculate tokens correctly using the shape of input_ids from the dict
    input_len = inputs["input_ids"].shape[1]
    generated = output_ids[0][input_len:]
    decoded_text = tokenizer.decode(generated, skip_special_tokens=True)
    usage = {
        "prompt_tokens": int(input_len),
        "completion_tokens": int(generated.shape[0]),
        "total_tokens": int(input_len + generated.shape[0]),
    }
    return {"text": decoded_text, "usage": usage}


def predict_openrouter(messages, model_id="google/gemma-2-27b-it", max_tokens=2048):
    import requests

    api_key = ENV_VARS.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = ENV_VARS.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")

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
            "temperature": 0,
        },
        timeout=180,
    )
    response.raise_for_status()
    payload = response.json()
    text = payload["choices"][0]["message"]["content"]
    usage = payload.get("usage", {})
    return {"text": text, "usage": usage}


def evaluate_model(model_name, model_id, predict_fn, test_df, few_shot_examples,
                   output_dir, seed):
    import concurrent.futures

    predictions = []
    ground_truths = []
    raw_responses = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0.0
    total_runtime = 0.0

    LOGGER.info("Evaluating %s on %d samples.", model_name, len(test_df))
    start_time = time.perf_counter()

    def process_row(args):
        idx, row = args
        text = _extract_row_text(row)
        gold = _extract_row_entities(row)
        messages = build_few_shot_messages(SYSTEM_PROMPT, few_shot_examples, text)
        sample_start = time.perf_counter()
        try:
            result = predict_fn(messages)
            parsed = parse_llm_response(result["text"])
        except Exception as exc:
            LOGGER.exception("Evaluation failed for %s sample %s: %s", model_name, idx, exc)
            parsed = {label: [] for label in LABELS}
            result = {"text": "", "usage": {}}
        elapsed = time.perf_counter() - sample_start
        return gold, parsed, result, elapsed

    max_workers = 1 if "krikri" in model_name.lower() else 10

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, (gold, parsed, result, elapsed) in enumerate(executor.map(process_row, test_df.iterrows())):
            predictions.append(parsed)
            ground_truths.append(gold)
            raw_responses.append(result["text"])
            total_runtime += elapsed

            prompt_tokens = int(result.get("usage", {}).get("prompt_tokens", 0) or 0)
            completion_tokens = int(result.get("usage", {}).get("completion_tokens", 0) or 0)
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens

            total_cost += estimate_api_cost(model_id, result.get("usage", {}))

            if i % 20 == 0 and i > 0:
                LOGGER.info("Processed %d/%d samples for %s.", i, len(test_df), model_name)

    metrics = compute_ner_metrics(predictions, ground_truths, LABELS)
    elapsed_total = time.perf_counter() - start_time
    gpu_hours = round(elapsed_total / 3600, 6)

    environment = collect_environment_metadata()
    run_configuration = collect_run_configuration(
        model_id=model_id,
        k=5,
        seed=seed,
        max_new_tokens=2048,
        temperature=0,
        do_sample=False,
        system_prompt=SYSTEM_PROMPT,
    )

    result_payload = {
        "model_name": model_name,
        "environment": environment,
        "run_configuration": run_configuration,
        "dataset_repo_id": REPO_ID,
        "dataset_revision": DATASET_REVISION,
        "sample_count": len(test_df),
        "labels": LABELS,
        "runtime_seconds": round(elapsed_total, 4),
        "gpu_hours": gpu_hours,
        "per_sample_runtime_seconds": round(total_runtime / max(len(test_df), 1), 6),
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "estimated_cost_usd": round(total_cost, 8),
        "metrics": metrics,
        "ground_truths": ground_truths,
        "predictions": predictions,
        "raw_responses": raw_responses,
    }

    output_path = output_dir / f"{model_name}_results.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result_payload, handle, ensure_ascii=False, indent=2)
    LOGGER.info("Saved results for %s to %s", model_name, output_path)
    LOGGER.info(
        "%s metrics: precision=%.4f recall=%.4f f1=%.4f gpu_hours=%.6f total_cost_usd=%.8f",
        model_name,
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        gpu_hours,
        total_cost,
    )

    qualitative_path = output_dir / f"{model_name}_qualitative.tsv"
    generate_qualitative_report(test_df, predictions, LABELS, qualitative_path)

    return result_payload


def print_results_table(results_dict):
    from tabulate import tabulate

    rows = []
    for name, payload in results_dict.items():
        metrics = payload["metrics"]
        rows.append([name, f"{metrics['precision']:.4f}", f"{metrics['recall']:.4f}", f"{metrics['f1']:.4f}"])

    print("\n" + tabulate(rows, headers=["Model", "Precision", "Recall", "F1"], tablefmt="grid"))

    for name, payload in results_dict.items():
        print(f"\n--- {name} per-label metrics ---")
        label_rows = []
        for label in LABELS:
            stats = payload["metrics"]["per_label"].get(label, {"precision": 0.0, "recall": 0.0, "f1": 0.0})
            label_rows.append([label, f"{stats['precision']:.4f}", f"{stats['recall']:.4f}", f"{stats['f1']:.4f}"])
        print(tabulate(label_rows, headers=["Label", "P", "R", "F1"], tablefmt="grid"))


def evaluate_models(n_samples=-1, model_name="all", output_dir=RESULTS_DIR, seed=42, shots_list=(0, 5), constrained=False):
    train_df, _, test_df = load_dataset_from_hf()
    
    if n_samples > 0:
        sample_df = test_df.head(n_samples).copy()
    else:
        sample_df = test_df.copy()
        
    if sample_df.empty:
        raise ValueError("No test samples available for evaluation.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = output_dir / f"llm_ner_extraction_{run_timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(log_file=str(run_dir / "execution.log"))
    LOGGER.info("Starting LLM NER extraction run in %s with shots %s", run_dir, shots_list)

    results = {}

    models_to_run = ["gemma", "llama", "qwen", "krikri"] if model_name == "all" else [model_name]

    for current_model in models_to_run:
        if current_model == "krikri":
            krikri_model, krikri_tokenizer = load_krikri_model()

        for shots in shots_list:
            few_shot_examples = select_few_shot_examples(train_df, k=shots, seed=seed) if shots > 0 else []

            if current_model == "gemma":
                def gemma_predict(messages):
                    return predict_openrouter(messages, model_id="google/gemma-4-26b-a4b-it")

                name = f"{shots}shot_gemma_4_26b"
                results[name] = evaluate_model(
                    name, "google/gemma-4-26b-a4b-it",
                    gemma_predict, sample_df, few_shot_examples, run_dir, seed=seed,
                )

            elif current_model == "llama":
                def llama_predict(messages):
                    return predict_openrouter(messages, model_id="meta-llama/llama-3.1-8b-instruct")

                name = f"{shots}shot_llama_3_1_8b"
                results[name] = evaluate_model(
                    name, "meta-llama/llama-3.1-8b-instruct",
                    llama_predict, sample_df, few_shot_examples, run_dir, seed=seed,
                )

            elif current_model == "qwen":
                def qwen_predict(messages):
                    return predict_openrouter(messages, model_id="qwen/qwen3-32b")

                name = f"{shots}shot_qwen3_32b"
                results[name] = evaluate_model(
                    name, "qwen/qwen3-32b",
                    qwen_predict, sample_df, few_shot_examples, run_dir, seed=seed,
                )

            elif current_model == "krikri":
                def krikri_predict(messages):
                    return predict_krikri(krikri_model, krikri_tokenizer, messages, constrained=constrained)

                name = f"{shots}shot_krikri_8b_constrained" if constrained else f"{shots}shot_krikri_8b"
                results[name] = evaluate_model(
                    name, "ilsp/Llama-Krikri-8B-Instruct-v1.5",
                    krikri_predict, sample_df, few_shot_examples, run_dir, seed=seed,
                )

        if current_model == "krikri":
            del krikri_model
            del krikri_tokenizer
            import torch
            torch.cuda.empty_cache()

    print_results_table(results)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run reproducible few-shot LLM baselines for Greek archaeological NER.")
    parser.add_argument("--samples", type=int, default=-1, help="Number of test examples to evaluate. Default is -1 (all samples).")
    parser.add_argument("--model", choices=["gemma", "llama", "qwen", "krikri", "all"], default="all", help="Select the model(s) to evaluate.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for few-shot example selection.")
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR), help="Directory for result artifacts.")
    parser.add_argument("--constrained", action="store_true", help="Use outlines to force JSON constraint (local models only).")
    return parser.parse_args()


def main():
    args = parse_args()
    samples_log = args.samples if args.samples > 0 else "all"
    LOGGER.info("Starting evaluation job: model=%s samples=%s seed=%d constrained=%s", args.model, samples_log, args.seed, args.constrained)
    evaluate_models(n_samples=args.samples, model_name=args.model, output_dir=args.output_dir, seed=args.seed, constrained=args.constrained)


if __name__ == "__main__":
    main()
