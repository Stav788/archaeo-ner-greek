#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset

# Add workspace to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gliner2 import GLiNER2
from gliner2.training.data import InputExample, TrainingDataset
from archaeo_ner_greek.training_utils import setup_local, df_to_gliner_examples

def run_significance():
    # 1. Environment Setup
    print("--- 🚀 Loading Environment & Configuration ---")
    env_vars = setup_local()
    
    # Paths
    base_dir = Path("/home/prokopis/src/archaeo-ner-greek")
    resources_dir = base_dir / "archaeo_ner_greek" / "resources"
    guidelines_path = resources_dir / "archaeoner_labels_definitions_v12_st.json"
    models_dir = base_dir / "data" / "models"
    
    # Load entity descriptions
    print(f"Loading labels from {guidelines_path}")
    with open(guidelines_path, 'r', encoding='utf-8') as f:
        entity_descriptions = json.load(f)
    
    # 2. Loading Dataset
    repo_id = env_vars.get("HF_REPO_ID", "Stalexan/archaeo-ner-greek")
    hf_token = env_vars.get("HF_TOKEN") or env_vars.get("HUGGING_FACE_HUB_TOKEN")
    print(f"Loading 'test' partition from HF: {repo_id}")
    ds = load_dataset(repo_id, name="default", token=hf_token)
    df_test = ds["test"].to_pandas()
    test_examples = df_to_gliner_examples(df_test, entity_descriptions)
    
    # 3. Model Definitions & Configurations
    models_info = {
        "Baseline (No-Synthetic)": {
            "adapter": models_dir / "gliner2_archaeo_lora_20260518_1704" / "best",
            "threshold": 0.8
        },
        "Old 500-Synthetic": {
            "adapter": models_dir / "gliner2_archaeo_lora_20260519_0101" / "best",
            "threshold": 0.5
        },
        "New Curated 1:1": {
            "adapter": models_dir / "gliner2_archaeo_lora_20260519_0256" / "best",
            "threshold": 0.8
        },
        "Medium-Capacity Baseline": {
            "adapter": models_dir / "gliner2_archaeo_lora_20260519_0822" / "best",
            "threshold": 0.7
        },
        "High-Capacity Baseline": {
            "adapter": models_dir / "gliner2_archaeo_lora_20260519_0704" / "best",
            "threshold": 0.8
        },
        "Real-Seeded (r=4)": {
            "adapter": models_dir / "gliner2_archaeo_lora_20260519_1206" / "best",
            "threshold": 0.8
        }
    }
    
    print("\n--- 🧠 Initializing GLiNER2 Model ---")
    device = "cpu"
    print(f"Using device: {device}")
    model = GLiNER2.from_pretrained("fastino/gliner2-multi-v1")
    model.to(device)
    
    # Collect sentence-level TP, FP, FN counts for each model
    sentence_counts = {name: [] for name in models_info.keys()}
    
    for name, config in models_info.items():
        print(f"\nEvaluating '{name}' with threshold {config['threshold']}...")
        adapter_path = config["adapter"]
        threshold = config["threshold"]
        
        # Load adapter
        model.load_adapter(str(adapter_path))
        model.eval()
        
        for i, ex in enumerate(test_examples):
            text = ex.text
            gt_entities = ex.entities
            
            # Predict
            output = model.extract_entities(text, entity_descriptions, threshold=threshold)
            pred_entities = output.get('entities', {})
            
            # Flatten spans
            pred_spans = [(t, lbl) for lbl, texts in pred_entities.items() for t in texts]
            gt_spans = [(t, lbl) for lbl, texts in gt_entities.items() for t in texts]
            
            # Calculate TP, FP, FN
            tp, fp, fn = 0, 0, 0
            temp_gt = gt_spans.copy()
            for p in pred_spans:
                if p in temp_gt:
                    tp += 1
                    temp_gt.remove(p)
                else:
                    fp += 1
            fn = len(temp_gt)
            
            sentence_counts[name].append({"tp": tp, "fp": fp, "fn": fn})
            
        # Print observed global performance
        tps = sum(c["tp"] for c in sentence_counts[name])
        fps = sum(c["fp"] for c in sentence_counts[name])
        fns = sum(c["fn"] for c in sentence_counts[name])
        precision = tps / (tps + fps) if (tps + fps) > 0 else 0
        recall = tps / (tps + fns) if (tps + fns) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"-> Observed: F1={f1:.4f} | P={precision:.4f} | R={recall:.4f} (TP={tps}, FP={fps}, FN={fns})")
    
    # 4. Bootstrap Resampling Significance Testing
    print("\n--- 📊 Running Paired Bootstrap Resampling (B=10,000) ---")
    np.random.seed(42)
    B = 10000
    N = len(test_examples)
    
    pairs = [
        ("Baseline (No-Synthetic)", "Real-Seeded (r=4)"),
        ("New Curated 1:1", "Real-Seeded (r=4)"),
        ("Old 500-Synthetic", "Real-Seeded (r=4)"),
        ("Baseline (No-Synthetic)", "New Curated 1:1"),
        ("Baseline (No-Synthetic)", "Old 500-Synthetic"),
        ("Baseline (No-Synthetic)", "Medium-Capacity Baseline"),
        ("Baseline (No-Synthetic)", "High-Capacity Baseline"),
        ("New Curated 1:1", "Old 500-Synthetic"),
        ("New Curated 1:1", "Medium-Capacity Baseline"),
        ("New Curated 1:1", "High-Capacity Baseline"),
        ("Medium-Capacity Baseline", "Old 500-Synthetic"),
        ("Medium-Capacity Baseline", "High-Capacity Baseline"),
        ("High-Capacity Baseline", "Old 500-Synthetic")
    ]
    
    significance_results = []
    
    for model_a_name, model_b_name in pairs:
        print(f"Testing difference: {model_b_name} vs. {model_a_name}")
        counts_a = sentence_counts[model_a_name]
        counts_b = sentence_counts[model_b_name]
        
        # Observed difference
        tps_a = sum(c["tp"] for c in counts_a)
        fps_a = sum(c["fp"] for c in counts_a)
        fns_a = sum(c["fn"] for c in counts_a)
        f1_a = 2 * (tps_a / (tps_a + fps_a)) * (tps_a / (tps_a + fns_a)) / ((tps_a / (tps_a + fps_a)) + (tps_a / (tps_a + fns_a)))
        
        tps_b = sum(c["tp"] for c in counts_b)
        fps_b = sum(c["fp"] for c in counts_b)
        fns_b = sum(c["fn"] for c in counts_b)
        f1_b = 2 * (tps_b / (tps_b + fps_b)) * (tps_b / (tps_b + fns_b)) / ((tps_b / (tps_b + fps_b)) + (tps_b / (tps_b + fns_b)))
        
        observed_diff = f1_b - f1_a
        print(f"  Observed Difference (F1_B - F1_A): {observed_diff:+.4f}")
        
        diffs = []
        for b in range(B):
            # Sample indices with replacement
            boot_idx = np.random.choice(N, size=N, replace=True)
            
            # Model A bootstrap metrics
            boot_tps_a = sum(counts_a[i]["tp"] for i in boot_idx)
            boot_fps_a = sum(counts_a[i]["fp"] for i in boot_idx)
            boot_fns_a = sum(counts_a[i]["fn"] for i in boot_idx)
            boot_prec_a = boot_tps_a / (boot_tps_a + boot_fps_a) if (boot_tps_a + boot_fps_a) > 0 else 0
            boot_rec_a = boot_tps_a / (boot_tps_a + boot_fns_a) if (boot_tps_a + boot_fns_a) > 0 else 0
            boot_f1_a = 2 * boot_prec_a * boot_rec_a / (boot_prec_a + boot_rec_a) if (boot_prec_a + boot_rec_a) > 0 else 0
            
            # Model B bootstrap metrics
            boot_tps_b = sum(counts_b[i]["tp"] for i in boot_idx)
            boot_fps_b = sum(counts_b[i]["fp"] for i in boot_idx)
            boot_fns_b = sum(counts_b[i]["fn"] for i in boot_idx)
            boot_prec_b = boot_tps_b / (boot_tps_b + boot_fps_b) if (boot_tps_b + boot_fps_b) > 0 else 0
            boot_rec_b = boot_tps_b / (boot_tps_b + boot_fns_b) if (boot_tps_b + boot_fns_b) > 0 else 0
            boot_f1_b = 2 * boot_prec_b * boot_rec_b / (boot_prec_b + boot_rec_b) if (boot_prec_b + boot_rec_b) > 0 else 0
            
            diffs.append(boot_f1_b - boot_f1_a)
            
        diffs = np.array(diffs)
        
        # Calculate p-value
        # One-sided test (H1: Model B > Model A)
        p_one_sided = np.sum(diffs <= 0) / B
        
        # Two-sided test (H1: Model B != Model A)
        p_two_sided = 2 * min(np.sum(diffs <= 0) / B, np.sum(diffs > 0) / B)
        
        print(f"  One-sided p-value (H1: Model 2 > Model 1) : {p_one_sided:.4f}")
        print(f"  Two-sided p-value (H1: Model 2 != Model 1): {p_two_sided:.4f}")
        
        significance_results.append({
            "comparison": f"{model_b_name} vs. {model_a_name}",
            "observed_diff": f"{observed_diff:+.4f}",
            "p_one_sided": p_one_sided,
            "p_two_sided": p_two_sided,
            "significant": "Yes" if p_two_sided < 0.05 else "No"
        })
        
    # 5. Clean / remove docs/significance_report.md
    legacy_report = base_dir / "docs" / "significance_report.md"
    if legacy_report.exists():
        print(f"🗑️ Deleting legacy report file: {legacy_report}")
        legacy_report.unlink()
        
    # 6. Update docs/TODO.md directly
    todo_path = base_dir / "docs" / "TODO.md"
    if todo_path.exists():
        print(f"📝 Updating {todo_path} with new benchmarks and significance p-values...")
        with open(todo_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        new_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Match start of benchmarks table
            if "## 📊 Unified Model Performance Benchmarks" in line:
                new_lines.append(line)
                new_lines.append("\n")
                # Write updated benchmarks table
                new_lines.append("| Model Name & Identifier | Training Dataset Composition | LoRA Config ($r / \\\\alpha$) | Trainable Parameters | Optimal Calibrated Threshold | Dev (Validation) Split Metrics <br> (Precision / Recall / F1) | Gold Test Set Metrics <br> (Precision / Recall / F1) | Raw Counts <br> (TP / FP / FN) |\n")
                new_lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
                new_lines.append("| **Standard Baseline** <br> `gliner2_archaeo_lora_20260518_1704` | `260` Human Sentences | $4$ / $8$ | `663,552` <br> *(0.25% of base)* | **`0.8`** | `0.7812` / `0.5097` / **`0.6172`** | `0.7386` / `0.5509` / **`0.6311`** | `65` / `23` / `53` |\n")
                new_lines.append("| **Old 500-Synthetic** <br> `gliner2_archaeo_lora_20260519_0101` | `260` Human + `500` Uncurated Synth | $4$ / $8$ | `663,552` <br> *(0.25% of base)* | **`0.5`** | `0.5957` / `0.5833` / **`0.5896`** | `0.6952` / `0.6186` / **`0.6547`** | `73` / `32` / `45` |\n")
                new_lines.append("| **New Curated 1:1** <br> `gliner2_archaeo_lora_20260519_0256` | `260` Human + `260` Curated Synth | $4$ / $8$ | `663,552` <br> *(0.25% of base)* | **`0.8`** | `0.7360` / `0.5679` / **`0.6406`** | `0.7368` / `0.5679` / **`0.6400`** | `67` / `24` / `51` |\n")
                new_lines.append("| **Medium-Capacity Baseline** <br> `gliner2_archaeo_lora_20260519_0822` | `260` Human Sentences | **$8$** / **$16$** | `1,327,104` <br> *(0.43% of base)* | **`0.7`** | `0.6929` / `0.6154` / **`0.6519`** | `0.6634` / `0.5678` / **`0.6119`** | `67` / `34` / `51` |\n")
                new_lines.append("| **High-Capacity Baseline** <br> `gliner2_archaeo_lora_20260519_0704` | `260` Human Sentences | **$16$** / **$32$** | `2,654,208` <br> *(1.01% of base)* | **`0.8`** | `0.7016` / `0.6084` / **`0.6517`** | `0.6800` / `0.5763` / **`0.6239`** | `68` / `32` / `50` |\n")
                new_lines.append("| **Real-Seeded (r=4)** <br> `gliner2_archaeo_lora_20260519_1206` | `260` Human + `58` Real-Seeded Synth | $4$ / $8$ | `663,552` <br> *(0.25% of base)* | **`0.8`** | `0.7236` / `0.6224` / **`0.6692`** | `0.7556` / `0.5763` / **`0.6538`** | `68` / `22` / `50` |\n")
                
                # Skip the old benchmarks table lines until next section
                i += 1
                while i < len(lines) and not lines[i].startswith("---") and not lines[i].startswith("## "):
                    i += 1
                continue
                
            # Match start of significance table
            if "## 🔬 Pairwise Statistical Significance (Bootstrap Resampling)" in line:
                new_lines.append(line)
                new_lines.append("\n")
                new_lines.append("Pairwise statistical significance test results using **Bootstrap Resampling** ($B=10,000$ draws) on the 32 unseen Gold Test sentences:\n\n")
                new_lines.append("| Comparison (Model 2 vs. Model 1) | Observed F1 Difference | One-sided p-value (Model 2 > Model 1) | Two-sided p-value (Model 2 != Model 1) | Statistically Significant (α = 0.05)? |\n")
                new_lines.append("| :--- | :---: | :---: | :---: | :---: |\n")
                for res in significance_results:
                    new_lines.append(f"| **{res['comparison'].split(' vs. ')[0]}** vs. **{res['comparison'].split(' vs. ')[1]}** | **{res['observed_diff']}** | `{res['p_one_sided']:.4f}` | `{res['p_two_sided']:.4f}` | **{res['significant']}** |\n")
                
                # Skip old table lines
                i += 1
                while i < len(lines) and not lines[i].startswith("---") and not lines[i].startswith("## "):
                    i += 1
                continue
                
            new_lines.append(line)
            i += 1
            
        with open(todo_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
            
        print("Success! docs/TODO.md has been fully updated.")

if __name__ == "__main__":
    run_significance()
