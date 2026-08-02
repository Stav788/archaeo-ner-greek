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
# # EDA on archaeo-ner-greek
#
# Analysis of the Archaeological NER dataset.

# %%
import os
import regex as re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datasets import load_dataset
from dotenv import load_dotenv, find_dotenv
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from archaeo_ner_greek.utils import get_project_root

# Setup logs directory for EDA PNG output artifacts using pathlib & get_project_root
project_root = get_project_root()
LOGS_DIR = Path(os.getenv("LOGS_DIR")) if os.getenv("LOGS_DIR") else (project_root / "logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Data Loading
# Fetching the `default` configuration partitions from Hugging Face.

# %%
load_dotenv(find_dotenv(), override=True)
repo_id = os.getenv("HF_REPO_ID")
hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
if not repo_id:
    raise ValueError("HF_REPO_ID is not set in the .env file.")

ds = load_dataset(repo_id, name="default", token=hf_token, revision="dev")

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
plt.savefig(LOGS_DIR / "entity_label_distribution.png", dpi=300)
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
    
    # Vocabulary & TTR calculation (word tokens)
    all_words = []
    for text in df['input']:
        all_words.extend(re.findall(r'\w+', text.lower()))
        
    total_words = len(all_words)
    vocab_size = len(set(all_words))
    ttr = (vocab_size / total_words * 100) if total_words > 0 else 0
    
    return {
        "Sentences": len(df),
        "Total Tokens": df['tokens'].sum(),
        "Avg. Length": df['tokens'].mean().round(2),
        "Vocabulary Size": vocab_size,
        "TTR (%)": round(ttr, 2),
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
# ## 3.1 Entity Span Metrics & Lexical Diversity
# Unique surface forms and average span lengths per entity label.

# %%
def get_entity_span_metrics(df):
    from collections import defaultdict
    span_data = defaultdict(list)
    
    for _, row in df.iterrows():
        text = row['input']
        labels = row.get('labels', [])
        for ent in labels:
            lbl = ent['label']
            start, end = ent['start'], ent['end']
            mention = text[start:end].strip()
            if mention:
                span_data[lbl].append(mention)
                
    results = []
    for lbl, mentions in span_data.items():
        total_m = len(mentions)
        unique_m = len(set(m.lower() for m in mentions))
        avg_tokens = np.mean([len(m.split()) for m in mentions])
        
        results.append({
            "Label": lbl,
            "Total Mentions": total_m,
            "Unique Surface Forms": unique_m,
            "Unique Ratio (%)": round((unique_m / total_m * 100), 2) if total_m > 0 else 0,
            "Avg. Span Length (Tokens)": round(avg_tokens, 2)
        })
        
    return pd.DataFrame(results).sort_values("Total Mentions", ascending=False)

df_span_metrics = get_entity_span_metrics(total_df)
print("\n--- Entity Span Metrics & Surface Form Diversity (Total Corpus) ---")
print(df_span_metrics.to_string(index=False))

# %% [markdown]
# ## 3.2 Accentuation System Distribution
# Monotonic vs. Polytonic Greek text distribution across the corpus.

# %%
def get_accentuation_metrics(df):
    polytonic_pattern = re.compile(r'[\u1f00-\u1ffe]')
    
    poly_count = 0
    mono_count = 0
    
    for text in df['input']:
        if polytonic_pattern.search(text):
            poly_count += 1
        else:
            mono_count += 1
            
    total = len(df)
    return {
        "Monotonic Sentences": mono_count,
        "Monotonic (%)": round((mono_count / total * 100), 2) if total > 0 else 0,
        "Polytonic Sentences": poly_count,
        "Polytonic (%)": round((poly_count / total * 100), 2) if total > 0 else 0,
    }

acc_metrics = get_accentuation_metrics(total_df)
print("\n--- Accentuation System Distribution (Total Corpus) ---")
for k, v in acc_metrics.items():
    print(f"{k}: {v}")


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
plt.savefig(LOGS_DIR / "semantic_distribution_tsne.png", dpi=300)
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

    # Crop to content (handle white background)
    from PIL import ImageOps
    
    # Invert to find non-white bounds
    inverted = ImageOps.invert(img)
    bbox = inverted.getbbox()
    
    if bbox:
        # Add padding to the content bounding box
        final_bbox = (
            max(0, bbox[0] - 20), 
            max(0, bbox[1] - 20), 
            min(img_width, bbox[2] + 20), 
            min(img_height, bbox[3] + 20)
        )
        img = img.crop(final_bbox)

    output_path = LOGS_DIR / f"viz_{filename}.png"
    img.save(output_path)
    print(f"Saved visualization to {output_path}")

# Example Export
for name, df in splits.items():
    if len(df) > 0:
        example = df.iloc[0]
        save_ner_png(example['input'], example['labels'], f"{name.lower()}_0")

# %% [markdown]
# ## 5. Annotation Consistency Audit
# Audits multi-label conflicts and repeated mention consistency across the corpus.

# %%
from archaeo_ner_greek.utils import audit_annotation_consistency

audit_results = audit_annotation_consistency(total_df)
print("\n--- Annotation Consistency Audit (Total Corpus) ---")
print(f"Overall Label Consistency Score: {audit_results['consistency_score']}%")
print(f"Total Unique Mentions: {audit_results['total_unique_mentions']}")
print(f"Total Repeated Mentions: {audit_results['total_repeated_mentions']}")
print(f"Consistent Repeated Mentions: {audit_results['consistent_repeated_mentions']}")
print(f"Conflicting Mentions Count: {audit_results['conflicting_mentions_count']}")

import unicodedata

if not audit_results['conflicts_df'].empty:
    print("\n--- ALL 27 Conflicting Surface Forms with Full Greek Sentences ---")
    latex_items = []
    
    for idx, r in audit_results['conflicts_df'].iterrows():
        mention = unicodedata.normalize('NFC', r['Mention'])
        total = r['Total Occurrences']
        breakdown = ", ".join([f"{k}: {v}" for k, v in r['Labels Breakdown'].items()])
        
        item_text = [f"\\subsubsection*{{\\foreignlanguage{{greek}}{{{mention}}} ({total} \\foreignlanguage{{greek}}{{εμφανίσεις}} --- {breakdown})}}", "\\begin{itemize}"]
        
        for lbl, items in r['Sentence Provenance'].items():
            for it in items:
                sent_txt = unicodedata.normalize('NFC', it['text'])
                doc_id = it['id']
                clean_sent = sent_txt.replace('&', r'\&').replace('%', r'\%').replace('#', r'\#')
                clean_doc_id = str(doc_id).replace('&', r'\&').replace('_', r'\_').replace('%', r'\%').replace('#', r'\#')
                item_text.append(f"  \\item \\textbf{{[{lbl}]}} (\\texttt{{{clean_doc_id}}}): \\foreignlanguage{{greek}}{{{clean_sent}}}")
                
        item_text.append("\\end{itemize}\n")
        latex_items.append("\n".join(item_text))
        
    full_latex_todo = "\n".join(latex_items)
    
    # Write to a dedicated file for paper_todo.tex inclusion
    todo_out_dir = project_root / "tmp" / "archaeoner_paper"
    todo_out_dir.mkdir(parents=True, exist_ok=True)
    todo_out_file = todo_out_dir / "todo_consistency_sentences.tex"
    with todo_out_file.open("w", encoding="utf-8") as f:
        f.write(full_latex_todo)
    print(f"✅ Saved complete Greek sentence audit list to {todo_out_file}")

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
