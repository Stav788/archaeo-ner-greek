import os
import sys
import subprocess
import logging
from pathlib import Path
from collections import defaultdict

# Verification Constants
VERIFICATION_TARGET_IDS = [
    "4d82e2e6-02c9-4fef-81f0-191ae553cb0f",
    "80811c3d-c530-4420-a660-da15a3459cfe",
    "2003_culture.gov_excavation_1371_10",
    "2025_nationalarchive_K7F58Y_598308_1"
]
VERIFICATION_TARGETS = ["αστρικό κόσμημα", "τριπτά εργαλεία"]
VERIFICATION_PAIR_TEXT = "παραστάδες"
VERIFICATION_PAIR_LABEL = "FEATURE"

# Project Constants
DEFAULT_ANNOTATOR = os.getenv("ANNOTATOR_A")

logger = logging.getLogger(__name__)

def setup_local():
    """Sets up local environment: loads .env variables."""
    from dotenv import dotenv_values, find_dotenv
    env_path = find_dotenv()
    return dotenv_values(env_path) if env_path else {}

def setup_colab():
    """Sets up Google Colab environment: installs deps, clones repo, and loads secrets."""
    logger.info("Pipeline Version: 1.2.9")
    logger.info(">>> Environment: Google Colab")
    from google.colab import userdata
    
    # 1. Install uv
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "uv"])

    GITHUB_TOKEN = userdata.get('GITHUB_TOKEN')
    REPO_NAME = "archaeo-ner-greek"
    REPO_URL = f"https://{GITHUB_TOKEN}@github.com/Stav788/{REPO_NAME}.git"
    
    if not os.path.exists(REPO_NAME):
        logger.info(f"Cloning repository: {REPO_URL} into {REPO_NAME}")
        subprocess.check_call(["git", "clone", "--branch", "dev", REPO_URL])
    
    REPO_PATH = Path(os.getcwd()) / REPO_NAME
    if str(REPO_PATH) not in sys.path:
        sys.path.append(str(REPO_PATH))
    
    os.chdir(str(REPO_PATH))

    # 2. Use uv to install the project and all dependencies into the system environment
    logger.info("Installing project dependencies via uv...")
    subprocess.check_call(["uv", "pip", "install", "--system", "-e", "."])
    
    # 3. Explicitly double-check critical missing libraries using the active interpreter's pip
    logger.info("Verifying critical libraries (mammoth, markdownify)...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "mammoth", "markdownify", "wtpsplit"])
    
    # Force Python to re-scan for the newly installed packages
    import importlib
    importlib.invalidate_caches()

    # Verification Step
    try:
        import mammoth
        import markdownify
        logger.info("Verification Successful: mammoth and markdownify are ready.")
    except ImportError as e:
        logger.error(f"Critical Verification Failed: {e}")
        raise

    def get_secret(key):
        try: return userdata.get(key)
        except: return None

    return {
        "ARGILLA_API_URL": get_secret("ARGILLA_API_URL"),
        "ARGILLA_API_KEY": get_secret("ARGILLA_API_KEY"),
        "ARGILLA_WORKSPACE": get_secret("ARGILLA_WORKSPACE"),
        "ARGILLA_DATASET": get_secret("ARGILLA_DATASET"),
        "ARGILLA_TEST_DATASET": get_secret("ARGILLA_TEST_DATASET"),
        "ANNOTATOR_A": get_secret("ANNOTATOR_A"),
    }

def df_to_gliner_examples(df, entity_descriptions):
    """Converts a DataFrame of annotations into a list of GLiNER2 InputExample objects."""
    from gliner2.training.data import InputExample
    examples = []
    for _, row in df.iterrows():
        text = row['sentence_field']
        entities = {lbl: [] for lbl in entity_descriptions.keys()}
        labels = row.get('labels', [])
        for label_obj in labels:
            lbl = label_obj['label']
            start = label_obj['start']
            end = label_obj['end']
            mention = text[start:end].strip()
            if lbl in entities:
                entities[lbl].append(mention)
        
        examples.append(InputExample(
            text=text,
            entities=entities,
            entity_descriptions=entity_descriptions,                
        ))
    return examples

def verify_annotations(df, target_ids, targets, pair_text, pair_label, annotator):
    """Debug function to verify specific problematic samples or targets."""
    for _, row in df.iterrows():
        text = row['sentence_field']
        labels = row.get('labels', [])
        record_id = str(row['id'])
        
        current_matches = []
        for label_obj in labels:
            lbl = label_obj['label']
            start = label_obj['start']
            end = label_obj['end']
            mention = text[start:end].strip()
            
            # Check targets
            for t in targets:
                if t.lower() in mention.lower():
                    current_matches.append(f"[FOUND] Entity: '{mention}' (Match: '{t}') | Label: {lbl}")

            # Check pair
            if pair_text.lower() in mention.lower() and lbl.upper() == pair_label.upper():
                current_matches.append(f"[FOUND] Pair: '{mention}' | Label: {lbl}")

        if current_matches:
            for m in current_matches:
                logger.debug(f"{m} | ID: {record_id}")
        elif record_id in target_ids:
            logger.debug(f"\n--- Debug for ID: {record_id} ---")
            logger.debug(f"Full Text: {text}")
            if not labels:
                logger.debug("Labels: [NONE]")
            for l in labels:
                m = text[l['start']:l['end']].strip()
                logger.debug(f"Actual Label by {annotator}: '{m}' as {l['label']}")

def plot_training_history(results):
    """Plots training and validation loss/metrics from GLiNER2Trainer results."""
    import matplotlib.pyplot as plt
    
    # 1. Plot Loss
    plt.figure(figsize=(10, 5))
    eval_hist = results.get('eval_metrics_history', [])
    
    # Eval Loss (usually what we have per epoch)
    if any('eval_loss' in x for x in eval_hist):
        plt.plot([x['eval_loss'] for x in eval_hist], label='Val Loss', marker='o')
    
    # Metrics
    plt.plot([x['f1'] for x in eval_hist], label='Val F1', marker='s')
    
    plt.title("Evaluation Progress (per Epoch)")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_threshold_curves(thresholds, p_scores, r_scores, f1_scores, default_threshold):
    """Plots Precision, Recall, and F1 vs. Threshold."""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Performance vs. Threshold
    plt.subplot(1, 2, 1)
    plt.plot(thresholds, p_scores, label='Precision', marker='s', color='#2ca02c')
    plt.plot(thresholds, r_scores, label='Recall', marker='o', color='#d62728')
    plt.plot(thresholds, f1_scores, label='F1 Score', marker='x', color='#1f77b4', linewidth=2)
    plt.axvline(x=default_threshold, color='gray', linestyle='--', label=f'Threshold ({default_threshold})')
    plt.title("Metrics vs. Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Precision-Recall Curve
    plt.subplot(1, 2, 2)
    plt.plot(r_scores, p_scores, marker='o', color='purple', linewidth=2)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def extract_doc_ids(df, id_column='document_sentence_id_field'):
    """Extracts parent document IDs by splitting at the last underscore."""
    return df[id_column].str.rsplit('_', n=1).str[0]

def grouped_split(df, group_col, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Original simple document-grouped split.
    Splits a DataFrame into three parts ensuring that all rows with the 
    same 'group_col' stay in the same split.
    """
    import random
    unique_groups = list(df[group_col].unique())
    random.seed(seed)
    random.shuffle(unique_groups)
    
    n = len(unique_groups)
    train_idx = int(n * train_ratio)
    val_idx = int(n * (train_ratio + val_ratio))
    
    train_groups = set(unique_groups[:train_idx])
    val_groups = set(unique_groups[train_idx:val_idx])
    test_groups = set(unique_groups[val_idx:])
    
    df_train = df[df[group_col].isin(train_groups)].copy()
    df_val = df[df[group_col].isin(val_groups)].copy()
    df_test = df[df[group_col].isin(test_groups)].copy()
    
    return df_train, df_val, df_test

def find_best_split_seeds(df, group_col, num_trials=100, top_n=5):
    """
    Tries multiple random seeds for grouped_split and returns the seeds 
    that result in the most balanced sample distributions (closest to 80/10/10).
    """
    results = []
    total_samples = len(df)
    target_ratios = [0.8, 0.1, 0.1]
    
    for seed in range(num_trials):
        df_train, df_val, df_test = grouped_split(df, group_col, seed=seed)
        
        # Calculate current ratios
        current_ratios = [
            len(df_train) / total_samples,
            len(df_val) / total_samples,
            len(df_test) / total_samples
        ]
        
        # Calculate Mean Squared Error from targets
        error = sum((a - b)**2 for a, b in zip(current_ratios, target_ratios))
        
        results.append({
            "seed": seed,
            "error": error,
            "counts": [len(df_train), len(df_val), len(df_test)],
            "ratios": current_ratios
        })
        
    # Sort by lowest error
    results.sort(key=lambda x: x["error"])
    return results[:top_n]

def plot_ner_confusion_matrix(model, dataset, entity_descriptions, threshold=0.8):
    """
    Plots a confusion matrix for NER results, including an 'O' category 
    for False Positives and False Negatives.
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    
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

def compute_metrics(model, dataset, threshold=0.8):
    """
    Calculates Micro-F1 and Per-Label PRF metrics.
    """
    import sys
    # Global counters
    tp, fp, fn = 0, 0, 0
    
    # Per-label counters
    label_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    
    model.eval()

    for i, ex in enumerate(dataset):
        text = ex[0]
        gt_entities = ex[1]["entities"]
        entity_descriptions = ex[1]["entity_descriptions"]

        output = model.extract_entities(text, entity_descriptions, threshold=threshold)
        pred_entities = output.get('entities', {})
        
        # Flatten pred spans: list of (text, label)
        pred_spans = [(t, lbl) for lbl, texts in pred_entities.items() for t in texts]
        # Flatten gt spans: list of (text, label)
        gt_spans = [(t, lbl) for lbl, texts in gt_entities.items() for t in texts]
        
        # 1. Calculate TPs and FPs
        temp_gt = gt_spans.copy()
        for p in pred_spans:
            text_p, lbl_p = p
            if p in temp_gt:
                tp += 1
                label_stats[lbl_p]["tp"] += 1
                temp_gt.remove(p)
            else:
                fp += 1
                label_stats[lbl_p]["fp"] += 1
        
        # 2. Calculate FNs (remaining in temp_gt)
        for g in temp_gt:
            text_g, lbl_g = g
            fn += 1
            label_stats[lbl_g]["fn"] += 1

    # Aggregate Global Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Aggregate Per-Label Metrics
    per_label_metrics = {}
    all_labels = set(list(label_stats.keys()) + list(dataset[0][1]["entities"].keys()))
    
    for lbl in all_labels:
        s = label_stats[lbl]
        l_p = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) > 0 else 0
        l_r = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) > 0 else 0
        l_f1 = 2 * (l_p * l_r) / (l_p + l_r) if (l_p + l_r) > 0 else 0
        per_label_metrics[lbl] = {
            "precision": l_p, "recall": l_r, "f1": l_f1, 
            "tp": s["tp"], "fp": s["fp"], "fn": s["fn"]
        }
    
    metrics = {
        "f1": f1, "precision": precision, "recall": recall, 
        "tp": tp, "fp": fp, "fn": fn,
        "per_label_metrics": per_label_metrics
    }
    
    # Log summary
    logger.info(f"\n>>> EVAL Micro-F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
    sys.stdout.flush()
    
    return metrics

def get_cnt(data):
    """Counts total entity mentions in a dataset or list of InputExamples."""
    exs = getattr(data, "examples", data)
    return sum(len(mentions) for ex in exs for mentions in ex.entities.values())

def evaluate_adapter(model, adapter_path, test_data, threshold=0.8):
    """
    Loads a specific LoRA adapter and evaluates its performance on the given dataset.
    """
    logger.info(f"Loading adapter from: {adapter_path}")
    model.load_adapter(adapter_path)
    
    test_data_formatted = [
        (ex.text, {"entities": ex.entities, "entity_descriptions": ex.entity_descriptions}) 
        for ex in test_data
    ]

    # Execute metric calculation
    results = compute_metrics(model, test_data_formatted, threshold=threshold)
    logger.info(f"\n--- EVALUATION RESULTS ({adapter_path.name}) threshold: {threshold} ---")
    logger.info(f"F1 Score : {results['f1']:.4f}")
    logger.info(f"Precision: {results['precision']:.4f}")
    logger.info(f"Recall   : {results['recall']:.4f}")
    logger.info(f"Counts   : TP={results['tp']}, FP={results['fp']}, FN={results['fn']}")
    
    return results

def show_error_analysis(model, dataset, entity_descriptions, threshold=0.8, num_examples=5):
    """Provides qualitative error analysis by highlighting TPs, FPs, and FNs in the console."""
    print(f"--- QUALITATIVE ERROR ANALYSIS (Threshold: {threshold}) ---\n")
    
    for i, ex in enumerate(dataset[:num_examples]):
        text, gt_entities = ex[0], ex[1]["entities"]
        
        # Get Predictions
        output = model.extract_entities(text, entity_descriptions, threshold=threshold)
        pred_entities = output.get('entities', {})
        
        # Flatten for comparison
        gt_spans = [(t, lbl) for lbl, texts in gt_entities.items() for t in texts]
        pred_spans = [(t, lbl) for lbl, texts in pred_entities.items() for t in texts]
        
        # Categorize
        tp = [p for p in pred_spans if p in gt_spans]
        fp = [p for p in pred_spans if p not in gt_spans]
        fn = [g for g in gt_spans if g not in pred_spans]
        
        # Display
        print(f"EXAMPLE {i+1}:")
        print(f"TEXT: {text[:150]}...")
        
        if tp: print(f"  [TPs]: {tp}")
        if fp: print(f"  \033[91m[FPs]: {fp}\033[0m") # Red
        if fn: print(f"  \033[93m[FNs]: {fn}\033[0m") # Yellow
        print("-" * 50)

def show_detailed_report(model, dataset, threshold=0.5):
    """
    Calculates per-category PRF metrics and displays a sortable table.
    """
    import pandas as pd
    from training_utils import compute_metrics
    
    # 1. Get predictions (compute_metrics returns per_label_metrics by default in our utils)
    results = compute_metrics(model, dataset, threshold=threshold)
    
    # Emoji Mapping
    emojis = {
        "ARTEFACT": "🏺",
        "LOCATION": "🏛️",
        "PERIOD": "⏳",
        "CONTEXT": "📜",
        "MATERIAL": "🧱",
        "SPECIES": "🧬",
        "PERSON": "👤",
        "FEATURE": "🗺️"
    }
    
    per_label = results.get('per_label_metrics', {})
    
    rows = []
    for label, metrics in per_label.items():
        emoji = emojis.get(label, "🏷️")
        rows.append({
            "Entity Type": f"{emoji} {label}",
            "Precision": float(metrics['precision']),
            "Recall": float(metrics['recall']),
            "F1-Score": float(metrics['f1']),
            "Support": int(metrics.get('tp', 0) + metrics.get('fn', 0))
        })
    
    if not rows:
        return "No entity predictions found for report."

    # Sort by F1 descending by default
    df = pd.DataFrame(rows).sort_values("F1-Score", ascending=False)
    
    # Add Micro Average Row
    micro_row = pd.DataFrame([{
        "Entity Type": "📊 OVERALL (Micro)",
        "Precision": float(results['precision']),
        "Recall": float(results['recall']),
        "F1-Score": float(results['f1']),
        "Support": sum(d['Support'] for d in rows)
    }])
    
    df = pd.concat([df, micro_row], ignore_index=True)
    
    # Format for display
    return df.style.format({
        "Precision": "{:.3f}",
        "Recall": "{:.3f}",
        "F1-Score": "{:.3f}"
    }).background_gradient(cmap='Blues', subset=['F1-Score'])

def safe_wandb_log(metrics, project, experiment_name, config=None, prefix=""):
    """
    Logs metrics to WandB safely, ensuring a run is active.
    """
    try:
        import wandb
        if wandb.run is None:
            # Handle dictionary vs TrainingConfig object
            config_dict = config.__dict__ if hasattr(config, "__dict__") else config
            wandb.init(project=project, name=experiment_name, config=config_dict, reinit="finish_previous")
        
        # Format metrics with prefix
        log_dict = {f"{prefix}{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}
        if log_dict:
            wandb.log(log_dict)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"WandB logging failed: {e}")

def setup_wandb(enabled, project, experiment_name, config):
    """
    Initializes WandB if enabled and returns a pre-configured logging function 
    to minimize code pollution in the main script.
    """
    if enabled:
        try:
            import wandb
            config_dict = config.__dict__ if hasattr(config, "__dict__") else config
            wandb.init(project=project, name=experiment_name, config=config_dict, reinit="finish_previous")
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"WandB initialization failed: {e}")

    def log_fn(metrics, prefix=""):
        if enabled:
            safe_wandb_log(metrics, project, experiment_name, config, prefix)
            
    return log_fn

def upload_wandb_artifact(enabled, adapter_path, experiment_name):
    """Uploads the best LoRA adapter as a WandB artifact if enabled."""
    if not enabled:
        return
    try:
        import wandb
        if wandb.run is None:
            import logging
            logging.getLogger(__name__).warning("No active WandB run found. Artifact upload skipped.")
            return

        artifact_name = f"adapter-{experiment_name}"
        artifact = wandb.Artifact(name=artifact_name, type="model")
        artifact.add_dir(str(adapter_path))
        wandb.log_artifact(artifact)
        
        import logging
        logging.getLogger(__name__).info(f"Adapter successfully uploaded to WandB: {artifact_name}")
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"WandB artifact upload failed: {e}")
