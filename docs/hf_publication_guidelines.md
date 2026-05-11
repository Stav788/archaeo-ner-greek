# Guidelines for Publishing Archaeological NER Datasets to Hugging Face

This guide outlines the steps for a user to publish the consolidated dataset to their own Hugging Face (HF) account using the provided pipeline.

## 1. Prerequisites & Authentication

- **HF Write Token**: Generate a token with **write** permissions at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
- **Environment Variables**: Add the token to your local `.env` file as `HF_TOKEN`.

## 2. Configuration

Ensure your `.env` contains the following target repository information:

- `HF_REPO_ID`: The full identifier for your dataset (e.g., `your-username/archaeo-ner-greek`).
- `HF_REPO_PRIVATE`: Set to `True` for private hosting, `False` for public.
- `HF_REPO_GATED`: Set to `manual` to require approval for access, or `false` to disable gating.
- `DEFAULT_ANNOTATOR`: Your specific Argilla username (the script will only publish your verified records).

## 3. Execution

Run the consolidated publication script from the project root:

```bash
uv run notebooks/publish_dataset_to_hf.py
```

## 4. Dataset Architecture (Automated)

The script automatically produces two subsets to satisfy both training and archival needs:

- **`argilla` subset**: A full archival backup containing every original record, column, and metadata field.
- **`default` subset**: A clean, stratified training set with:
  - **GLiNER2 Compatibility**: Standardized `input` (text) and `output` (entity dictionary) columns.
  - **Stratified Splits**: Grouped by document ID (80/10/10) to ensure rare labels appear in every partition without data leakage.
  - **Precision Preservation**: Original character-level offsets are kept in the `labels` column for auditing.

## 5. Verification & Governance

- **Integrity Check**: The script performs an automatic "round-trip" verification, pulling the data back from HF and comparing record IDs and counts against the in-memory source.
- **Access Control**: If `HF_REPO_GATED` is set, the script automatically configures the repository settings on the Hub.
- **Contact Metadata**: The script updates the DatasetCard (`README.md`) with the `HF_NOTIFICATION_EMAIL` specified in your `.env`.
