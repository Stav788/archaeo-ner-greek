import os
import sys
import subprocess
import logging
from pathlib import Path
from gliner2.training.data import InputExample

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

logger = logging.getLogger(__name__)

def setup_local():
    """Sets up local environment: loads .env variables."""
    from dotenv import dotenv_values, find_dotenv
    env_path = find_dotenv()
    return dotenv_values(env_path) if env_path else {}

def setup_colab():
    """Sets up Google Colab environment: installs deps, clones repo, and loads secrets."""
    logger.info(">>> Environment: Google Colab")
    from google.colab import userdata
    
    # Install dependencies
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "gliner2", "argilla", "tabulate", "python-dotenv",
                           "matplotlib", "seaborn", "scikit-learn", "mammoth", "markdownify", "wtpsplit"])

    GITHUB_TOKEN = userdata.get('GITHUB_TOKEN')
    REPO_NAME = "archaeo-ner-greek"
    REPO_URL = f"https://{GITHUB_TOKEN}@github.com/Stav788/{REPO_NAME}.git"
    
    if not os.path.exists(REPO_NAME):
        logger.info(f"Cloning repository: {REPO_URL} into {REPO_NAME}")
        subprocess.check_call(["git", "clone", "--branch", "dev", REPO_URL])
    
    REPO_PATH = Path(os.getcwd()) / REPO_NAME
    if str(REPO_PATH) not in sys.path:
        sys.path.append(str(REPO_PATH))
    
    if not REPO_PATH.is_dir():
        raise FileNotFoundError(f"Repository directory not found at {REPO_PATH}")
    
    os.chdir(str(REPO_PATH))

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
    plt.plot([x['loss'] for x in results.history if 'loss' in x], label='Train Loss')
    if any('eval_loss' in x for x in results.history):
        plt.plot([x['eval_loss'] for x in results.history if 'eval_loss' in x], label='Val Loss')
    plt.title("Training Progress (Loss)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
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
    Splits a DataFrame into three parts (train, val, test) ensuring that all 
    rows with the same 'group_col' value stay in the same split.
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
    Bulleted Micro-F1 calculation with local path-based schema loading.
    """
    import sys
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
