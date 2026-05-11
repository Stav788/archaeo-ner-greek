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
# # Inter-Annotator Agreement (IAA) Analysis
#
# This script calculates agreement metrics between annotators using data pulled 
# exclusively from the Hugging Face `argilla` subset.
#
# ## Metrics Calculated:
# 1. **Pairwise Agreement**: Global sentence-level match % and Span F1-Score.
# 2. **Label Breakdown**: Per-label F1-scores to identify ambiguous categories.
# 3. **Discrepancy Analysis**: Boundary disagreements vs. classification errors.
# 4. **Annotator Distribution**: Comparison of label usage density.

# %%
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from datasets import load_dataset
from archaeo_ner_greek.utils import (
    prepare_iaa_data, 
    calculate_ner_iaa, 
    calculate_label_iaa, 
    analyze_iaa_discrepancies
)
from archaeo_ner_greek.logging_config import setup_logging
import logging

# %%
# 1. Setup logging and environment
setup_logging()
logger = logging.getLogger(__name__)
load_dotenv()

# Configuration from .env
repo_id = os.getenv("HF_REPO_ID")
hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
annotator_a = os.getenv("ANNOTATOR_A")
annotator_b = os.getenv("ANNOTATOR_B")

if not repo_id:
    logger.error("HF_REPO_ID not found in .env file.")
    raise ValueError("Missing HF_REPO_ID")

logger.info(f"Analyzing IAA for: {repo_id}")
logger.info(f"Annotator A: {annotator_a}")
logger.info(f"Annotator B: {annotator_b}")

# %% [markdown]
# ## 2. Data Loading (from Hugging Face)
# We pull the `argilla` subset which contains the full archival record with responses.

# %%
logger.info(f"Fetching 'argilla' subset from {repo_id}...")
dataset = load_dataset(repo_id, name="argilla", split="train", token=hf_token)
df = dataset.to_pandas()

logger.info(f"Loaded {len(df)} records from HF.")

# %% [markdown]
# ## 3. Structuring Data for IAA
# We transform the raw responses into (start, end, label) triplets.

# %%
# Prepare data for agreement analysis
iaa_results = prepare_iaa_data(df)
iaa_ready = iaa_results["ready"]

# Report annotation status
print(f"\n--- Annotation Status ---")
print(f"✅ Sentences ready for IAA (2+ annotators): {len(iaa_ready)}")
print(f"⚠️  Missing teammates (only 1 annotator): {len(iaa_results['missing_teammate'])}")
print(f"❌ Unannotated sentences: {len(iaa_results['unannotated'])}")

# %% [markdown]
# ## 4. Global Pairwise Agreement
# Calculating high-level consistency metrics.

# %%
print(f"\n--- Global Agreement Report ---")
iaa_report = calculate_ner_iaa(iaa_ready)
print(iaa_report.to_string(index=False))

# %% [markdown]
# ## 5. Per-Label Agreement Breakdown
# Identifying which entity types cause the most confusion.

# %%
print(f"\n--- Per-Label Agreement Breakdown ---")
label_report = calculate_label_iaa(iaa_ready, annotators=[annotator_a, annotator_b])
print(label_report.to_string(index=False))

# %% [markdown]
# ## 6. Discrepancy Analysis
# Quantifying boundary vs. classification errors.

# %%
print(f"\n--- Discrepancy Analysis ---")
discrepancy_report = analyze_iaa_discrepancies(iaa_ready, annotators=[annotator_a, annotator_b])
print(discrepancy_report)

# %% [markdown]
# ## 7. Visualizations & Final Reporting
# Ported from legacy notebook for comprehensive thesis metrics.

# %%
# 1. Set display options for better inspection
pd.set_option('display.max_colwidth', None)

# 2. Comprehensive Visualization Dashboard
def plot_iaa_dashboard(label_report, discrepancy_report, annotator_a, annotator_b):
    # Set visual style
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 18))
    
    # --- Panel 1: Label F1-Scores ---
    label_data_clean = label_report[label_report["Label"] != "OVERALL"].sort_values("F1-Score", ascending=False)
    sns.barplot(data=label_data_clean, x="F1-Score", y="Label", ax=axes[0], palette="viridis")
    axes[0].set_title("IAA F1-Score per Label", fontsize=14, fontweight='bold')
    axes[0].set_xlim(0, 1)
    
    # --- Panel 2: Agreement vs Disagreement Volume ---
    # Prepare data for grouped bar chart
    col_a = f"Only in A ({annotator_a})"
    col_b = f"Only in B ({annotator_b})"
    label_counts = label_data_clean.melt(
        id_vars="Label", 
        value_vars=["Both", col_a, col_b], 
        var_name="Category", 
        value_name="Count"
    )
    sns.barplot(data=label_counts, x="Count", y="Label", hue="Category", ax=axes[1], palette="muted")
    axes[1].set_title("Annotation Volume Breakdown (Agreement vs Omissions)", fontsize=14, fontweight='bold')
    axes[1].legend(title="Source", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # --- Panel 3: Discrepancy Types Distribution ---
    discrepancy_data = discrepancy_report.reset_index()
    discrepancy_data.columns = ["Error Type", "Count"]
    sns.barplot(data=discrepancy_data, x="Count", y="Error Type", ax=axes[2], palette="rocket")
    axes[2].set_title("Distribution of Discrepancy Types", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

# 3. Execution
if len(iaa_ready) > 0:
    plot_iaa_dashboard(label_report, discrepancy_report, annotator_a, annotator_b)
    
    print("\n--- FINAL TABLES ---")
    print("\n[GLOBAL AGREEMENT]")
    print(iaa_report.to_string(index=False))
    print("\n[LABEL BREAKDOWN]")
    print(label_report.to_string(index=False))
    print("\n[DISCREPANCY SUMMARY]")
    print(discrepancy_report)
else:
    print("No data ready for visualization.")
