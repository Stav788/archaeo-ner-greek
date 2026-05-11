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
# # EDA - Archaeological NER
#
# Analysis of the Archaeological NER dataset.

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
pivot_stats['Total'] = pivot_stats.sum(axis=1)
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

# Add 'Total' collection row
total_df = pd.concat(splits.values())
m_total = get_corpus_metrics(total_df)
m_total_row = pd.DataFrame([m_total], index=['Total'])
df_metrics = pd.concat([df_metrics, m_total_row])

print("--- Corpus Metrics (Splits & Total) ---")
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
# Using spaCy's `displacy` to generate SVG renderings.

# %%
from PIL import Image, ImageDraw, ImageFont

def save_ner_png(text, entities, filename):
    """
    Renders NER entities using PIL and saves as PNG for thesis inclusion.
    """
    if entities is None or len(entities) == 0:
        print(f"No entities found for {filename}, skipping.")
        return

    # Archaeological Color Palette (RGB)
    COLORS = {
        "ARTEFACT": (122, 236, 236),
        "LOCATION": (255, 149, 97),
        "PERIOD": (170, 156, 252),
        "MATERIAL": (255, 235, 128),
        "SIGHT": (156, 201, 204),
        "PERSON": (228, 255, 135),
        "ORGANIZATION": (255, 129, 151),
        "SPECIES": (189, 147, 249),
        "TXT": (40, 40, 40),
        "BG": (255, 255, 255)
    }

    font_size = 32
    label_size = 20
    
    # Load fonts (Standard Linux paths)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", label_size)
    except:
        font = ImageFont.load_default()
        label_font = ImageFont.load_default()

    # Create canvas
    img_width, img_height = 4000, 400
    img = Image.new("RGB", (img_width, img_height), color=COLORS["BG"])
    draw = ImageDraw.Draw(img)
    
    current_x, current_y = 50, 100
    last_idx = 0
    padding_h, padding_v = 12, 10
    
    # Process entities
    for ent in sorted(entities, key=lambda x: x['start']):
        start, end, label = ent['start'], ent['end'], ent['label']
        
        # 1. Plain text before
        pre_text = text[last_idx:start]
        if pre_text:
            draw.text((current_x, current_y), pre_text, font=font, fill=COLORS["TXT"])
            current_x += draw.textlength(pre_text, font=font)
            
        # 2. Entity box
        ent_text = text[start:end]
        ent_width = draw.textlength(ent_text, font=font)
        label_text = f" {label}"
        label_width = draw.textlength(label_text, font=label_font)
        total_box_width = ent_width + label_width
        
        box_coords = [
            current_x - padding_h, 
            current_y - padding_v, 
            current_x + total_box_width + padding_h, 
            current_y + font_size + padding_v
        ]
        fill_color = COLORS.get(label, (220, 220, 220))
        draw.rounded_rectangle(box_coords, radius=10, fill=fill_color)
        
        # Draw entity word
        draw.text((current_x, current_y), ent_text, font=font, fill=COLORS["TXT"])
        # Draw label
        draw.text((current_x + ent_width + 6, current_y + 10), label, font=label_font, fill=COLORS["TXT"])
        
        current_x += total_box_width + (padding_h * 2) + 15
        last_idx = end

    # 3. Remaining text
    post_text = text[last_idx:]
    if post_text:
        draw.text((current_x, current_y), post_text, font=font, fill=COLORS["TXT"])

    # Crop to content
    bbox = img.getbbox()
    if bbox:
        # Add padding to the content bounding box
        final_bbox = (
            max(0, bbox[0] - 20), 
            max(0, bbox[1] - 20), 
            min(img_width, bbox[2] + 20), 
            min(img_height, bbox[3] + 20)
        )
        img = img.crop(final_bbox)

    output_path = f"viz_{filename}.png"
    img.save(output_path)
    print(f"Saved visualization to {output_path}")

# Example Export
for name, df in splits.items():
    if len(df) > 0:
        example = df.iloc[0]
        save_ner_png(example['input'], example['labels'], f"{name.lower()}_0")

# %%
print("\nEDA for Archaeological NER metrics complete.")

# %% [markdown]
# ## 6. Export to LaTeX
# Generates thesis-ready tables if EXPORT_LATEX_TABLES is enabled.

# %%
if os.getenv("EXPORT_LATEX_TABLES", "False").lower() == "true":
    from archaeo_ner_greek.utils import export_to_latex_table
    
    # 1. Entity Distribution
    export_to_latex_table(
        pivot_stats.reset_index(), 
        caption="Entity label distribution per split.", 
        label="tab:eda-entity-distribution"
    )
    
    # 2. Corpus Metrics
    export_to_latex_table(
        df_metrics.reset_index(), 
        caption="Corpus statistics and averages per split.", 
        label="tab:eda-corpus-metrics"
    )
