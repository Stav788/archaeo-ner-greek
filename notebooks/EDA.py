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
# # Exploratory Data Analysis (EDA) - Archaeological NER
#
# This notebook provides a comprehensive analysis of the Archaeological NER dataset.
#
# **Key Requirements:**
# 1. Use ONLY the Hugging Face `default` dataset subset.
# 2. Detailed split-level metrics (Train/Val/Test).
# 3. Semantic similarity analysis of splits via embeddings.
# 4. XeLaTeX-compatible annotation export.

# %%
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv, find_dotenv
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer

# %% [markdown]
# ## 1. Data Loading
# Fetching the `default` configuration partitions from Hugging Face.

# %%
load_dotenv(find_dotenv(), override=True)
repo_id = os.getenv("HF_REPO_ID", "pprokopidis/archaeo-ner-greek")
hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

print(f"Loading partitions from: {repo_id}")
ds = load_dataset(repo_id, name="default", token=hf_token)

# Map splits to DataFrames
splits = {
    "Train": ds["train"].to_pandas(),
    "Validation": ds["validation"].to_pandas(),
    "Test": ds["test"].to_pandas()
}

for name, df in splits.items():
    print(f"{name} split: {len(df)} records")

# %% [markdown]
# ## 2. Per-Split Entity Distributions
# Comparing label representation across partitions.

# %%
def get_label_stats(df):
    all_labels = []
    for labels_list in df['labels']:
        for ent in labels_list:
            all_labels.append(ent['label'])
    
    counts = pd.Series(all_labels).value_counts()
    pct = (counts / counts.sum() * 100).round(2)
    return pd.DataFrame({"Count": counts, "Pct": pct})

# Aggregate stats for all splits
combined_stats = []
for name, df in splits.items():
    stats = get_label_stats(df)
    stats['Split'] = name
    combined_stats.append(stats.reset_index().rename(columns={'index': 'Label'}))

df_stats = pd.concat(combined_stats)

# Visualization
plt.figure(figsize=(12, 6))
sns.set_theme(style="whitegrid")
sns.barplot(data=df_stats, x="Label", y="Count", hue="Split", palette="muted")
plt.title("Entity Label Distribution per Split", fontweight='bold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Display Table
print("--- Detailed Label Distribution ---")
pivot_stats = df_stats.pivot(index='Label', columns='Split', values='Count').fillna(0).astype(int)
print(pivot_stats)

# %% [markdown]
# ## 3. Corpus Statistics & Averages
# Sentence lengths and entity density per split.

# %%
def get_corpus_metrics(df):
    df = df.copy()
    df['tokens'] = df['input'].apply(lambda x: len(x.split()))
    df['ents'] = df['labels'].apply(len)
    
    return {
        "Sentences": len(df),
        "Total Tokens": df['tokens'].sum(),
        "Avg. Length": df['tokens'].mean().round(2),
        "Total Entities": df['ents'].sum(),
        "Avg. Entities/Sent": df['ents'].mean().round(2),
        "Density (Ent/100 Tokens)": (df['ents'].sum() / df['tokens'].sum() * 100).round(2)
    }

metrics_list = []
for name, df in splits.items():
    m = get_corpus_metrics(df)
    m['Split'] = name
    metrics_list.append(m)

df_metrics = pd.DataFrame(metrics_list).set_index('Split')
print("--- Per-Split Corpus Metrics ---")
print(df_metrics)

# %% [markdown]
# ## 4. Semantic Similarity & Clustering
# Checking if Train/Val/Test constitute similar semantic clusters.

# %%
print("Calculating semantic distribution...")
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    all_texts = []
    split_labels = []
    for name, df in splits.items():
        all_texts.extend(df['input'].tolist())
        split_labels.extend([name] * len(df))
    
    embeddings = model.encode(all_texts, show_progress_bar=True)
    method = "S-BERT Embeddings"
except ImportError:
    print("sentence-transformers not found. Falling back to TF-IDF for similarity check.")
    vectorizer = TfidfVectorizer(max_features=1000)
    all_texts = []
    split_labels = []
    for name, df in splits.items():
        all_texts.extend(df['input'].tolist())
        split_labels.extend([name] * len(df))
    embeddings = vectorizer.fit_transform(all_texts).toarray()
    method = "TF-IDF Features"

# Dimensionality Reduction with T-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
projections = tsne.fit_transform(embeddings)

# Plotting
plt.figure(figsize=(10, 7))
sns.scatterplot(
    x=projections[:, 0], y=projections[:, 1], 
    hue=split_labels, style=split_labels, 
    alpha=0.6, palette="deep"
)
plt.title(f"Semantic Distribution of Splits ({method})", fontweight='bold')
plt.xlabel("T-SNE 1")
plt.ylabel("T-SNE 2")
plt.legend(title="Split")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Entity Visualization Export
# Using spaCy's `displacy` to generate high-quality SVG renderings for the thesis.

# %%
from spacy import displacy

def save_displacy_svg(text, entities, filename):
    """
    Renders NER entities using displacy and saves the output as an SVG file.
    """
    if entities is None or len(entities) == 0:
        print(f"No entities found for {filename}, skipping.")
        return

    # Prepare data for displacy manual mode
    # ents must be sorted by start position
    formatted_ents = [
        {"start": e['start'], "end": e['end'], "label": e['label']} 
        for e in sorted(entities, key=lambda x: x['start'])
    ]
    
    doc_data = {
        "text": text,
        "ents": formatted_ents,
        "title": None
    }
    
    # Render to SVG
    # page=False returns just the SVG snippet
    svg = displacy.render(doc_data, style="ent", manual=True, jupyter=False, page=True)
    
    # Save to file
    output_path = f"viz_{filename}.svg"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"Saved visualization to {output_path}")

# Example Export: Save the first record of each split
for name, df in splits.items():
    if len(df) > 0:
        example = df.iloc[0]
        save_displacy_svg(example['input'], example['labels'], f"{name.lower()}_0")

# %%
print("\nEDA for Archaeological NER metrics complete.")
