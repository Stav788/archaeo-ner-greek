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
    
    # 3. Model Definitions & Configurations from Registry
    registry_path = base_dir / "data" / "models" / "model_registry.json"
    print(f"Loading model configurations from registry: {registry_path}")
    with open(registry_path, "r", encoding="utf-8") as f:
        registry = json.load(f)
        
    active_models = {
        "baseline (r=4)": "gliner2_archaeo_lora_20260518_1704",
        "augmented-unfiltered (r=4, n=500)": "gliner2_archaeo_lora_20260519_0101",
        "augmented-filtered (r=4, n=260)": "gliner2_archaeo_lora_20260519_0256",
        "baseline (r=8)": "gliner2_archaeo_lora_20260519_0822",
        "baseline (r=16)": "gliner2_archaeo_lora_20260519_0704",
        "augmented-seeded-strict (r=4, n=93)": "gliner2_archaeo_lora_20260519_2016"
    }
    
    models_info = {}
    for display, m_id in active_models.items():
        if m_id in registry:
            models_info[display] = {
                "adapter": models_dir / m_id / "best",
                "threshold": registry[m_id]["optimal_threshold"]
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
        ("baseline (r=4)", "augmented-seeded-strict (r=4, n=93)"),
        ("augmented-filtered (r=4, n=260)", "augmented-seeded-strict (r=4, n=93)"),
        ("augmented-unfiltered (r=4, n=500)", "augmented-seeded-strict (r=4, n=93)"),
        ("baseline (r=4)", "augmented-filtered (r=4, n=260)"),
        ("baseline (r=4)", "augmented-unfiltered (r=4, n=500)"),
        ("baseline (r=4)", "baseline (r=8)"),
        ("baseline (r=4)", "baseline (r=16)"),
        ("augmented-filtered (r=4, n=260)", "augmented-unfiltered (r=4, n=500)"),
        ("augmented-filtered (r=4, n=260)", "baseline (r=8)"),
        ("augmented-filtered (r=4, n=260)", "baseline (r=16)"),
        ("baseline (r=8)", "augmented-unfiltered (r=4, n=500)"),
        ("baseline (r=8)", "baseline (r=16)"),
        ("baseline (r=16)", "augmented-unfiltered (r=4, n=500)")
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
        print(f"📝 Updating {todo_path} with new benchmarks, significance p-values, and training history...")
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
                new_lines.append("| Model Name & Identifier | Training Dataset Composition | LoRA Config ($r / \\\\alpha$) | Trainable Parameters | Optimal Calibrated Threshold | Dev (Validation) Split Metrics <br> (Precision / Recall / F1) | Gold Test Set Metrics <br> (Precision / Recall / F1) | Raw Counts <br> (TP / FP / FN) |\n")
                new_lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
                
                active_order = [
                    "gliner2_archaeo_lora_20260518_1704",
                    "gliner2_archaeo_lora_20260519_0101",
                    "gliner2_archaeo_lora_20260519_0256",
                    "gliner2_archaeo_lora_20260519_0822",
                    "gliner2_archaeo_lora_20260519_0704",
                    "gliner2_archaeo_lora_20260519_2016"
                ]
                
                for m_id in active_order:
                    if m_id in registry:
                        m = registry[m_id]
                        dev_p, dev_r, dev_f = m["dev_metrics"]["precision"], m["dev_metrics"]["recall"], m["dev_metrics"]["f1"]
                        test_p, test_r, test_f = m["gold_test_metrics"]["precision"], m["gold_test_metrics"]["recall"], m["gold_test_metrics"]["f1"]
                        tp, fp, fn = m["gold_test_metrics"]["tp"], m["gold_test_metrics"]["fp"], m["gold_test_metrics"]["fn"]
                        
                        emoji = " 🚀" if m_id == "gliner2_archaeo_lora_20260519_2016" else ""
                        
                        params = m["trainable_parameters"]
                        pct = "*(0.25% of base)*"
                        if m_id == "gliner2_archaeo_lora_20260519_0822":
                            pct = "*(0.43% of base)*"
                        elif m_id == "gliner2_archaeo_lora_20260519_0704":
                            pct = "*(1.01% of base)*"
                        
                        lora_display = f"${m['lora_rank']}$ / ${m['lora_alpha']}$"
                        if "baseline" in m["display_name"] and m["lora_rank"] > 4:
                            lora_display = f"**{lora_display}**"
                            
                        # Format dynamic dataset name
                        comp = m["dataset_composition"]
                        parts = comp.split(" + ")
                        formatted_parts = []
                        for part in parts:
                            words = part.split(" ")
                            formatted_parts.append(f"`{words[0]}` " + " ".join(words[1:]))
                        formatted_comp = " + ".join(formatted_parts)
                            
                        new_lines.append(
                            f"| **{m['display_name']}** <br> [`{m_id}`]({m['wandb_url']}) | "
                            f"{formatted_comp} | "
                            f"{lora_display} | "
                            f"`{params}` <br> {pct} | "
                            f"**`{m['optimal_threshold']}`** | "
                            f"`{dev_p:.4f}` / `{dev_r:.4f}` / **`{dev_f:.4f}`** | "
                            f"`{test_p:.4f}` / `{test_r:.4f}` / **`{test_f:.4f}`**{emoji} | "
                            f"`{tp}` / `{fp}` / `{fn}` |\n"
                        )
                
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
                
                i += 1
                while i < len(lines) and not lines[i].startswith("---") and not lines[i].startswith("## "):
                    i += 1
                continue
                
            # Match start of hyperparameters & convergence table
            if "## 📈 Hyperparameters & Convergence Benchmarks" in line:
                new_lines.append(line)
                new_lines.append("\n")
                new_lines.append("This table tracks hyperparameters, training datasets, and peak validation metrics achieved during active model training prior to post-training inference calibration:\n\n")
                
                columns = [
                    "gliner2_archaeo_lora_20260518_1704",
                    "gliner2_archaeo_lora_20260519_0101",
                    "gliner2_archaeo_lora_20260519_0256",
                    "gliner2_archaeo_lora_20260519_0822",
                    "gliner2_archaeo_lora_20260519_0704",
                    "gliner2_archaeo_lora_20260519_1206",
                    "gliner2_archaeo_lora_20260519_1828",
                    "gliner2_archaeo_lora_20260519_2016"
                ]
                
                header_names = []
                for m_id in columns:
                    if m_id in registry:
                        header_names.append(registry[m_id]["display_name"])
                    else:
                        header_names.append(m_id)
                
                header = "| Metric / Parameter | " + " | ".join(header_names) + " |"
                align = "| :--- | " + " | ".join(":---:" for _ in columns) + " |"
                new_lines.append(header + "\n")
                new_lines.append(align + "\n")
                
                rank_row = "| **LoRA Rank ($r$)** | "
                for m_id in columns:
                    if m_id in registry:
                        m = registry[m_id]
                        rank_str = f"${m['lora_rank']}$"
                        if m['lora_rank'] in (8, 16):
                            rank_str = f"**{rank_str}**"
                        if m_id == "gliner2_archaeo_lora_20260519_0704":
                            rank_str += " *(Capacity bump)*"
                        rank_row += rank_str + " | "
                new_lines.append(rank_row.strip() + "\n")
                
                alpha_row = "| **LoRA Alpha ($\\alpha$)** | "
                for m_id in columns:
                    if m_id in registry:
                        m = registry[m_id]
                        alpha_str = f"${m['lora_alpha']}$"
                        if m['lora_alpha'] in (16, 32):
                            alpha_str = f"**{alpha_str}**"
                        alpha_row += alpha_str + " | "
                new_lines.append(alpha_row.strip() + "\n")
                
                size_row = "| **Dataset Size (Sentences)**| "
                for m_id in columns:
                    if m_id in registry:
                        m = registry[m_id]
                        comp = m["dataset_composition"]
                        parts = comp.split(" + ")
                        formatted_parts = []
                        for part in parts:
                            words = part.split(" ")
                            formatted_parts.append(f"`{words[0]}` ({words[1]})")
                        size_row += " + ".join(formatted_parts) + " | "
                new_lines.append(size_row.strip() + "\n")
                
                epoch_row = "| **Best Epoch** | "
                for m_id in columns:
                    if m_id in registry:
                        m = registry[m_id]
                        epoch_row += f"{m['training_history']['best_epoch']} | "
                new_lines.append(epoch_row.strip() + "\n")
                
                f1_row = "| **Peak Dev F1** | "
                for m_id in columns:
                    if m_id in registry:
                        m = registry[m_id]
                        f1 = m['training_history']['peak_dev_f1']
                        f1_str = f"`{f1:.4f}`"
                        if m_id == "gliner2_archaeo_lora_20260519_1206":
                            f1_str = f"**{f1_str}** 🚀 *(All-time Peak)*"
                        elif m_id == "gliner2_archaeo_lora_20260519_0822":
                            f1_str = f"**{f1_str}**"
                        f1_row += f1_str + " | "
                new_lines.append(f1_row.strip() + "\n")
                
                p_row = "| Peak Dev Precision | "
                for m_id in columns:
                    if m_id in registry:
                        m = registry[m_id]
                        p = m['training_history']['peak_dev_precision']
                        p_str = f"`{p:.4f}`"
                        if m_id == "gliner2_archaeo_lora_20260518_1704":
                            p_str = f"**{p_str}**"
                        p_row += p_str + " | "
                new_lines.append(p_row.strip() + "\n")
                
                r_row = "| Peak Dev Recall | "
                for m_id in columns:
                    if m_id in registry:
                        m = registry[m_id]
                        r = m['training_history']['peak_dev_recall']
                        r_str = f"`{r:.4f}`"
                        if m_id == "gliner2_archaeo_lora_20260519_1828":
                            r_str = f"**{r_str}** 🚀 *(Peak recall)*"
                        elif m_id == "gliner2_archaeo_lora_20260519_0822":
                            r_str = f"**{r_str}**"
                        r_row += r_str + " | "
                new_lines.append(r_row.strip() + "\n")
                
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
