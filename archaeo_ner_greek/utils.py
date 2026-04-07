import logging
import os
import ast
import mammoth
import markdownify
import pandas as pd
import argilla as rg
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Literal, Optional, Set, Union, Any
from pydantic import BaseModel
from dotenv import load_dotenv, dotenv_values, find_dotenv
from huggingface_hub import HfApi, login
from wtpsplit import SaT

logger = logging.getLogger(__name__)

# --- Templates ---

css_template = """
<div id="docs_content"></div>
<script id="docs_template" type="text/x-handlebars-template">
    <style>
    #container {
        display: flex;
        gap: 10px;
    }
    .column {
        flex: 1;
    }
    /* --- Added color rules --- */
    .column:nth-child(1) h3 {
        color: #4A90E2; /* Blue */
    }
    .column:nth-child(2) h3 {
        color: #2ECC71; /* Green */
    }
    .column:nth-child(3) h3 {
        color: #F39C12; /* Orange */
    }
    .column:nth-child(4) h3 {
        color: #9B59B6; /* Purple */
    }    
    </style>
    <div id="container">
        <div class="column">
            <h3>Source document</h3>
            <div>{{{record.fields.docs.source_doc}}}</div>
        </div>
        <div class="column">
            <h3>Ground truth</h3>
            <div>{{{record.fields.docs.ground_truth_doc}}}</div>
        </div>
        <div class="column">
            <h3>MT</h3>
            <div>{{{record.fields.docs.mt_doc}}}</div>
        </div>
        <div class="column">
            <h3>MT + Adaptation</h3>
            <div>{{{record.fields.docs.mt_adapt_doc}}}</div>
        </div>
    </div>
</script>    
"""

script = """
<script src="https://cdn.jsdelivr.net/npm/handlebars@latest/dist/handlebars.js"></script>
<script>
    const docs_template = document.getElementById("docs_template").innerHTML;
    const compiledTemplate = Handlebars.compile(docs_template);
    const html = compiledTemplate({ record });
    document.getElementById("docs_content").innerHTML = html;
</script>
"""

# --- Models ---

class UserConfig(BaseModel):
    username: str
    password: str
    role: Literal["owner", "admin", "annotator"]

# --- Credential Management ---

def load_credentials_from_env() -> Dict[str, str]:
    """
    Locates the .env file using find_dotenv and loads environment variables.
    """
    env_path = find_dotenv(usecwd=True, raise_error_if_not_found=False)

    if not env_path:
        logger.error(f"Configuration file missing at: {Path.cwd() / '.env'}") 
        return {}

    env_path_obj = Path(env_path)
    
    if not env_path_obj.exists():
        logger.error(f"Configuration file missing at: {env_path_obj}")
        return {}
        
    return dotenv_values(env_path_obj)

def connect_hugging_face(env_values: Optional[dict] = None):
    """
    Authenticates with Hugging Face Hub and verifies identity.
    """
    if env_values is None:
        env_values = load_credentials_from_env()

    hf_token = env_values.get("HUGGING_FACE_HUB_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    
    if not hf_token:
        logger.warning("HF Token not found in configuration.")
        return None

    try:
        # Login to local machine
        login(token=hf_token, add_to_git_credential=False)
        
        # Verify identity via API
        hf_api = HfApi(token=hf_token)
        user_info = hf_api.whoami()
        # logger.info(f"HF Authentication Successful: Logged in as '{user_info.get('name')}'")
        return hf_api
    except Exception as e:
        logger.error(f"HF Authentication failed: {e}")
        return None

# --- Argilla Client & User Management ---

def get_argilla_client(env_vars: Optional[Union[dict, str]] = None, env_path: Optional[str] = None) -> rg.Argilla:
    """
    Initializes and validates the Argilla client connection.
    
    Args:
        env_vars: Optional dictionary containing ARGILLA_API_URL and ARGILLA_API_KEY.
                  Can also be the env_path string for backward compatibility.
        env_path: Optional path to a .env file to load variables from.
    """
    # 1. Robust argument handling (handles stale notebook signatures)
    if isinstance(env_vars, str):
        env_path = env_vars
        env_vars = None
    elif isinstance(env_path, dict) and env_vars is None:
        # If someone calls get_argilla_client(env_path=my_dict)
        env_vars = env_path
        env_path = None
    
    # 2. Load from environment if no dict provided
    if env_vars is None:
        load_dotenv(dotenv_path=env_path)
        env_vars = {
            "ARGILLA_API_URL": os.getenv("ARGILLA_API_URL"),
            "ARGILLA_API_KEY": os.getenv("ARGILLA_API_KEY")
        }

    api_url = env_vars.get("ARGILLA_API_URL")
    api_key = env_vars.get("ARGILLA_API_KEY")

    if not api_url or not api_key:
        logger.error("Missing ARGILLA_API_URL or ARGILLA_API_KEY.")
        raise ValueError("Cannot initialize Argilla client due to missing configuration.")

    try:
        client = rg.Argilla(
            api_url=api_url,
            api_key=api_key
        )
        
        # Verify connectivity
        user_info = client.me
        # logger.info(f"Successfully connected to Argilla as user: {user_info.username}")
        
        return client
        
    except Exception as e:
        logger.error(f"Failed to establish connection with Argilla: {e}")
        raise e

# Alias for backward compatibility
configure_argilla_client = get_argilla_client

def ensure_argilla_users(
    client: rg.Argilla,
    users_config: List[UserConfig],
) -> None:
    """
    Ensures that the specified users exist in the Argilla instance.
    """
    valid_roles = {"owner", "admin", "annotator"}

    logger.info(f"Processing {len(users_config)} user configurations...")

    for user_data in users_config:
        username = user_data.username
        password = user_data.password
        role = user_data.role.lower()

        if not username or not password:
            logger.warning(f"Skipping entry with missing credentials: {user_data}")
            continue

        if role not in valid_roles:
            logger.warning(f"Role '{role}' is invalid for user '{username}'. Defaulting to 'annotator'.")
            role = "annotator"

        try:
            existing_user = client.users(username)
            if existing_user:
                logger.info(f"User '{username}' already exists. Skipping creation.")
                continue

            logger.info(f"Creating new user '{username}' with role '{role}'...")
            
            new_user = rg.User(
                username=username,
                password=password,
                role=role,
                first_name=username,
                last_name="(Bot/User)" 
            )
            
            new_user.create()
            logger.info(f"Successfully created user '{username}'.")

        except Exception as e:
            logger.error(f"Failed to manage user '{username}': {e}")

def configure_argilla_resources(client: rg.Argilla, env_vars: dict):
    """
    Configures Argilla users and workspaces based on environment variables.
    """
    usernames = env_vars.get("ARGIILA_USERNAMES", "").split(",")
    real_names = env_vars.get("ARGIILA_USERS", "").split(",")
    passwords = env_vars.get("ARGILLA_PASSWORDS", "").split(",")
    roles = env_vars.get("ARGILLA_ROLES", "").split(",")
    
    workspace_config_str = env_vars.get("ARGILLA_WORKSPACES", "[]")
    try:
        workspaces_config = ast.literal_eval(workspace_config_str)
    except (ValueError, SyntaxError):
        logger.error("Failed to parse ARGILLA_WORKSPACES. Ensure it is a valid list literal.")
        workspaces_config = []

    # 1. Process Users
    for username, real_name, password, role in zip(usernames, real_names, passwords, roles):
        logger.info(f"Processing user: {username}")
        if not client.users(username):
            try:
                user_to_create = rg.User(username=username, password=password)
                user_to_create.create()
                logger.info(f"User '{username}' created successfully.")
            except Exception:
                logger.warning(f"Could not create user '{username}'. They may already exist.", exc_info=True)

        try:
            user_to_update = client.users(username)
            name_parts = real_name.strip().split(" ", 1)
            first_name = name_parts[0]
            last_name = name_parts[1] if len(name_parts) > 1 else ""

            user_to_update.first_name = first_name
            user_to_update.last_name = last_name
            user_to_update.role = role
            user_to_update.update()
            logger.info(f"User '{username}' updated successfully.")
        except Exception:
            logger.error(f"Failed to update details for user '{username}'.", exc_info=True)

    # 2. Process Workspaces
    for workspace_data in workspaces_config:
        workspace_name = workspace_data.get("name")
        if not client.workspaces(workspace_name):
            try:
                workspace_to_create = rg.Workspace(name=workspace_name)
                workspace_to_create.create()
                logger.info(f"Workspace '{workspace_name}' created.")
            except Exception:
                logger.error(f"Failed to create workspace '{workspace_name}'.", exc_info=True)

        users_to_add = workspace_data.get("users_to_add", [])
        for username_to_add in users_to_add:
            try:
                user_obj = client.users(username_to_add)
                workspace_obj = client.workspaces(workspace_name)
                if user_obj not in workspace_obj.users:
                    user_obj.add_to_workspace(workspace_obj)
                logger.info(f"Added user '{username_to_add}' to workspace '{workspace_name}'.")
            except Exception:
                logger.error(f"Failed to add user '{username_to_add}' to workspace '{workspace_name}'.", exc_info=True)

def list_datasets(
    client: rg.Argilla, 
    dataset_name: Optional[str] = None, 
    workspace_name: Optional[str] = None, 
    username: Optional[str] = None,
    exact: bool = True
) -> List[rg.Dataset]:
    """
    Lists all datasets in Argilla with optional filtering.
    
    Args:
        client: Authenticated Argilla client.
        dataset_name: Filter by dataset name.
        workspace_name: Filter by workspace name.
        username: Optional username to filter by.
        exact: If True (default), use exact case-insensitive match. 
               If False, use partial substring match.
    """
    datasets = list(client.datasets)
    
    # 1. Filter by Username (Workspaces the user belongs to)
    if username:
        user = client.users(username)
        if not user:
            logger.warning(f"User '{username}' not found. Returning empty list.")
            return []
        
        # Get names of workspaces this user has access to
        user_workspaces = {ws.name for ws in user.workspaces}
        datasets = [d for d in datasets if d.workspace.name in user_workspaces]

    # 2. Filter by Workspace Name
    if workspace_name:
        if exact:
            datasets = [d for d in datasets if workspace_name.lower() == d.workspace.name.lower()]
        else:
            datasets = [d for d in datasets if workspace_name.lower() in d.workspace.name.lower()]

    # 3. Filter by Dataset Name
    if dataset_name:
        if exact:
            datasets = [d for d in datasets if dataset_name.lower() == d.name.lower()]
        else:
            datasets = [d for d in datasets if dataset_name.lower() in d.name.lower()]

    return datasets

def get_dataset(
    client: rg.Argilla, 
    dataset_name: str, 
    workspace_name: Optional[str] = None
) -> Optional[rg.Dataset]:
    """
    Returns the first Argilla dataset that matches the name exactly (case-insensitive).
    
    Args:
        client: Authenticated Argilla client.
        dataset_name: Name of the dataset to find.
        workspace_name: Optional workspace name filter.
    """
    datasets = list_datasets(client, dataset_name=dataset_name, workspace_name=workspace_name, exact=True)
    return datasets[0] if datasets else None

def get_dataset_users(dataset: rg.Dataset) -> List[rg.User]:
    """
    Returns a list of users who have access to the workspace of the specified dataset.
    
    Args:
        dataset: The Argilla Dataset object.
    """
    workspace = dataset.workspace
    if not workspace:
        logger.error(f"Workspace for dataset '{dataset.name}' not found.")
        return []
        
    # In Argilla v2, users associated with a workspace are accessible via workspace.users
    return list(workspace.users)

# --- Dataset Management ---

def setup_translation_dataset(client: rg.Argilla, dataset_name: str, workspace_name: str, guidelines: str, min_submitted = 2):
    """
    Recreates the Argilla dataset schema for translation/evaluation tasks.
    """
    try:
        existing_dataset = client.datasets(name=dataset_name, workspace=workspace_name)
        if existing_dataset:
            logger.info(f"Deleting existing dataset: {dataset_name}")
            existing_dataset.delete()
    except Exception:
        pass
        
    try:
        settings = rg.Settings(
            guidelines=guidelines,
            fields=[
                rg.CustomField(name="docs", template=css_template + script, advanced_mode=True),
                rg.TextField(name="id_field", title="ID"),
            ],
            distribution=rg.TaskDistribution(min_submitted=min_submitted),
            metadata=[
                rg.TermsMetadataProperty(name="id_meta", title="ID", visible_for_annotators=True),
            ],
            questions=[
                rg.RatingQuestion(name="accuracy_adequacy", title="Accuracy and Adequacy", values=[1, 2, 3, 4, 5]),
                rg.RatingQuestion(name="fluency_and_grammaticality", title="Fluency and Grammaticality", values=[1, 2, 3, 4, 5]),
                rg.RatingQuestion(name="cohesion_and_consistency", title="Cohesion and Consistency", values=[1, 2, 3, 4, 5]),
                rg.RatingQuestion(name="adaptation", title="Adaptation", values=[1, 2, 3, 4, 5]),
            ],
        )        

        dataset = rg.Dataset(name=dataset_name, workspace=workspace_name, settings=settings, client=client)
        dataset.create()
        logger.info(f"Translation dataset {dataset_name} created successfully.")
        return dataset
    except Exception:
        logger.error("Failed to setup translation dataset", exc_info=True)
        return None

def setup_ner_dataset(client: rg.Argilla, dataset_name: str, workspace_name: str, guidelines: str, min_submitted: int = 2):
    """
    Recreates the Argilla dataset schema for NER tasks.
    """
    try:
        existing_dataset = client.datasets(name=dataset_name, workspace=workspace_name)
        if existing_dataset:
            logger.info(f"Deleting existing dataset: {dataset_name}")
            existing_dataset.delete()
    except Exception:
        pass
        
    try:
        settings = rg.Settings(
            guidelines=guidelines,
            fields=[
                rg.TextField(name="sentence_field", title="Context Description", required=True),
            ],
            metadata=[
                rg.IntegerMetadataProperty(name="context_sheet_description_id", title="Context Sheet Description ID", visible_for_annotators=True),
            ],
            questions=[
                rg.SpanQuestion(
                    name="entities",
                    title="Entities",
                    field="sentence_field",
                    labels=[
                        'ARTEFACT', 'PERIOD', "LOCATION", 'CONTEXT', 'CONTEXT_ID', 
                        'MATERIAL', 'SPECIES', "FEATURE", "PERSON", "MISC"
                    ],
                    allow_overlapping=True,
                ),
                rg.TextQuestion(
                    name="label_suggestion",
                    title="Label suggestion",
                    description="Suggest a new label for MISC items.",
                    required=False
                ),
            ],
            distribution=rg.TaskDistribution(min_submitted=min_submitted)
        )
        
        dataset = rg.Dataset(name=dataset_name, workspace=workspace_name, settings=settings, client=client)
        dataset.create()
        logger.info(f"NER dataset {dataset_name} created successfully.")
        return dataset
    except Exception:
        logger.error("Failed to setup NER dataset", exc_info=True)
        return None

# Alias for backward compatibility
def setup_argilla_dataset(client: rg.Argilla, dataset_name: str, workspace_name: str, guidelines: str, min_submitted = 2):
    return setup_translation_dataset(client, dataset_name, workspace_name, guidelines, min_submitted)

def get_dataset_as_dataframe(
    client: rg.Argilla, 
    dataset_name: Union[str, rg.Dataset], 
    workspace_name: Optional[str] = None,
    include_responses: bool = False
) -> pd.DataFrame:
    """
    Exports an Argilla dataset to a pandas DataFrame.

    Args:
        client: Authenticated Argilla client.
        dataset_name: Name of the dataset or the Dataset object itself.
        workspace_name: Workspace name (required if dataset_name is a string and not unique).
        include_responses: If True, includes a column 'responses' with a list of dictionaries 
                           containing {'username': ..., 'values': ..., 'status': ...}.
    """
    try:
        if isinstance(dataset_name, str):
            dataset = client.datasets(name=dataset_name, workspace=workspace_name)
        else:
            dataset = dataset_name

        if not dataset:
            logger.error(f"Dataset '{dataset_name}' not found.")
            return pd.DataFrame()
        
        records = list(dataset.records)
        if not records:
            logger.warning(f"Dataset '{dataset.name}' is empty.")
            return pd.DataFrame()
            
        # Cache users for mapping user_id to username if needed
        user_map = {}
        if include_responses:
            try:
                user_map = {str(u.id): u.username for u in client.users}
            except Exception as e:
                logger.warning(f"Could not fetch user list for mapping IDs to names: {e}")

        data = []
        for r in records:
            row = {"id": r.id}
            row.update(r.fields)
            row.update(r.metadata)
            
            if include_responses:
                # Format: [{'username': 'prokopis', 'values': {...}, 'status': 'submitted'}, ...]
                row["responses"] = []
                for resp in r.responses:
                    # logger.info(dir(resp))
                    # In v2, response has user_id, not a user object
                    u_id = str(resp.user_id) if resp.user_id else None
                    row["responses"].append({
                        "username": user_map.get(u_id, "unknown") if u_id else "unknown",
                        "user_id": u_id,
                        "values": resp.value,
                        "status": resp.status
                    })
            
            data.append(row)
        return pd.DataFrame(data)
    except Exception:
        logger.error("Failed to export dataset to DataFrame.", exc_info=True)
        return pd.DataFrame()

def add_records_safely(client: rg.Argilla, dataset_name: str, workspace: str, new_records: List[rg.Record], batch_size: int = 500):
    """
    Uploads records to Argilla in batches.
    """
    dataset = client.datasets(name=dataset_name, workspace=workspace)
    if not dataset:
        logger.error(f"Dataset {dataset_name} not found.")
        return
    
    total = len(new_records)
    for i in range(0, total, batch_size):
        batch = new_records[i : i + batch_size]
        try:
            dataset.records.log(batch)
            logger.info(f"Logged batch {i//batch_size + 1}: {len(batch)} records ({min(i+batch_size, total)}/{total}).")
        except Exception as e:
            logger.error(f"Failed to log batch starting at index {i}: {e}")

def duplicate_dataset(
    client: rg.Argilla,
    source_dataset_name: str,
    source_workspace: str,
    target_dataset_name: str,
    target_workspace: str,
    usernames: Optional[List[str]] = None,
    min_submitted: Optional[int] = None,
    guidelines: Optional[str] = None,
    **kwargs
) -> None:
    """
    Clones an entire dataset (Settings + Records) to a target workspace.
    """
    try:
        source_ds = client.datasets(name=source_dataset_name, workspace=source_workspace)
        if not source_ds:
            logger.error(f"Source dataset '{source_dataset_name}' not found.")
            return

        # Ensure Target Workspace Exists
        target_ws = client.workspaces(target_workspace)
        if not target_ws:
            target_ws = rg.Workspace(name=target_workspace)
            target_ws.create()

        # Assign Users to Target Workspace
        if usernames or kwargs.get("users_to_add"):
            users_list = usernames or kwargs.get("users_to_add", [])
            for username in users_list:
                user = client.users(username)
                if user:
                    try:
                        user.add_to_workspace(target_ws)
                    except Exception:
                        pass

        # Copy Settings
        source_settings = source_ds.settings
        target_settings = rg.Settings(
            fields=source_settings.fields,
            questions=source_settings.questions,
            metadata=source_settings.metadata,
            vectors=source_settings.vectors,
            guidelines=guidelines if guidelines else source_settings.guidelines,
            allow_extra_metadata=source_settings.allow_extra_metadata,
            distribution=rg.TaskDistribution(min_submitted=min_submitted) if min_submitted else source_settings.distribution
        )

        # Handle existing target dataset
        existing_target = client.datasets(name=target_dataset_name, workspace=target_workspace)
        if existing_target:
            existing_target.delete()

        target_ds = rg.Dataset(
            name=target_dataset_name,
            workspace=target_workspace,
            settings=target_settings,
            client=client
        )
        target_ds.create()

        # Copy Records
        records_buffer = []
        BATCH_SIZE = 1000
        for rec in source_ds.records:
            new_rec = rg.Record(
                fields=rec.fields,
                metadata=rec.metadata,
                vectors=rec.vectors,
                suggestions=rec.suggestions,
                responses=rec.responses,
                id=rec.id
            )
            records_buffer.append(new_rec)
            if len(records_buffer) >= BATCH_SIZE:
                target_ds.records.log(records_buffer)
                records_buffer = []

        if records_buffer:
            target_ds.records.log(records_buffer)

        logger.info(f"Successfully duplicated dataset '{source_dataset_name}' to '{target_dataset_name}'.")
    except Exception as e:
        logger.error(f"Failed to duplicate dataset: {e}", exc_info=True)

# Alias for backward compatibility
copy_dataset = duplicate_dataset

# --- Collaborative Annotation Utils ---

def ensure_list(data: Any) -> List[Any]:
    """
    Safely converts data to a list. Handles stringified lists and NaNs.
    """
    if data is None:
        return []
    if isinstance(data, list):
        return data
    if pd.isna(data):
        return []
    if isinstance(data, str):
        s = data.strip()
        if not s or s.lower() == 'nan' or s.lower() == 'none':
            return []
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return []
    return []

def export_responses_as_suggestions(
    client: rg.Argilla,
    dataset_name: str,
    workspace: str
) -> pd.DataFrame:
    """
    Exports records where responses exist and formats them as suggestions.
    """
    try:
        dataset = client.datasets(name=dataset_name, workspace=workspace)
        if dataset is None:
            logger.error(f"Dataset '{dataset_name}' not found.")
            return pd.DataFrame()

        export_data = []
        for record in dataset.records:
            # Find responses that have been 'submitted'
            responses = [r for r in record.responses if r.status == "submitted"]
            
            if responses:
                # Take the most recent response
                target_response = responses[-1]
                entities_payload = target_response.value
                
                if isinstance(entities_payload, dict):
                    entities_value = entities_payload.get("value")
                else:
                    entities_value = entities_payload

                formatted_suggestion = [{
                    "question_name": "entities",
                    "value": entities_value,
                    "agent": f"user_response_{target_response.user_id}",
                    "score": 1.0
                }]

                row = {
                    "id": record.id,
                    "sentence_field": record.fields.get("sentence_field") or record.fields.get("text"),
                    "metadata": record.metadata,
                    "suggestions": formatted_suggestion 
                }
                export_data.append(row)

        return pd.DataFrame(export_data)
    except Exception as e:
        logger.error(f"Failed to export responses: {e}", exc_info=True)
        return pd.DataFrame()

def reimport_as_collaborative_dataset(
    client: rg.Argilla,
    df: pd.DataFrame,
    new_dataset_name: str,
    target_workspace: str,
    guidelines: str,
    min_submitted: int = 2
):
    """
    Re-imports a DataFrame as a collaborative NER dataset with suggestions.
    """
    if df.empty:
        logger.warning("DataFrame is empty. No dataset created.")
        return

    try:
        # Re-use setup_ner_dataset for schema consistency
        dataset = setup_ner_dataset(client, new_dataset_name, target_workspace, guidelines, min_submitted)
        
        records_to_log = []
        for _, row in df.iterrows():
            text_content = row.get("sentence_field") or row.get("text")
            if not text_content: continue
            
            record_suggestions = []
            raw_suggs = ensure_list(row.get("suggestions"))                    
            
            for s_dict in raw_suggs:
                if isinstance(s_dict, dict) and s_dict.get("question_name") == "entities":
                    record_suggestions.append(rg.Suggestion(
                        question_name="entities",
                        value=s_dict.get("value")
                    ))

            rec = rg.Record(
                fields={"sentence_field": text_content},
                metadata=row.get("metadata", {}),
                suggestions=record_suggestions,
                id=row.get("id")
            )
            records_to_log.append(rec)
            
        if records_to_log:
            add_records_safely(client, new_dataset_name, target_workspace, records_to_log)

    except Exception as e:
        logger.error(f"Failed to re-import dataset: {e}", exc_info=True)

# --- File Utils ---

def docx_to_markdown(docx_path: Union[str, Path], md_path: Union[str, Path]):
    """
    Converts a DOCX file to Markdown.
    """
    with open(docx_path, "rb") as docx_file:
        result = mammoth.convert_to_html(docx_file)
        html_content = result.value

    markdown_content = markdownify.markdownify(html_content, heading_style="ATX")

    with open(md_path, "w", encoding="utf-8") as md_file:
        md_file.write(markdown_content)
    
    logger.info(f"Converted {docx_path} to {md_path}")

def process_and_upload_greek_texts(
    client: rg.Argilla,
    dataset_name: str,
    workspace: str,
    source_dir: Path,
    context_window_size: int = 2,
    exclude_ids: Optional[Set[str]] = None
):
    """
    Processes Greek text files into Argilla records with context windows.
    """
    logger.info("Initializing SaT model (sat-3l, style=ud, lang=el)...")
    try:
        model = SaT("sat-3l", style_or_domain="ud", language="el")
    except Exception as e:
        logger.error(f"Failed to load SaT model: {e}")
        return

    try:
        dataset = client.datasets(name=dataset_name, workspace=workspace)
        current_global_count = len(list(dataset.records)) if dataset else 0
    except Exception:
        current_global_count = 0

    new_records_buffer = []
    if exclude_ids is None:
        exclude_ids = set()

    files = list(source_dir.glob("*.txt"))
    logger.info(f"Scanning {len(files)} text files in {source_dir}...")

    for file_path in files:
        try:
            basename = file_path.stem 
            if basename in exclude_ids:
                continue

            with open(file_path, "r", encoding="utf-8") as f:
                text_content = f.read()

            if not text_content.strip():
                continue

            sentences = [s.strip() for s in model.split(text_content) if s.strip()]
            
            for i, current_sentence in enumerate(sentences):
                doc_sent_id = f"{basename}_{i}"
                
                start = max(0, i - context_window_size)
                prev_text = " ".join(sentences[start:i]) if i > 0 else "---"

                end = min(len(sentences), i + 1 + context_window_size)
                next_text = " ".join(sentences[i+1:end]) if i < len(sentences) - 1 else "---"

                record = rg.Record(
                    fields={
                        "prev_sentences_field": prev_text,
                        "sentence_field": current_sentence,
                        "next_sentences_field": next_text,
                        "document_sentence_id_field": doc_sent_id
                    },
                    metadata={"sentence_id_metadata": current_global_count},
                    id=doc_sent_id
                )
                new_records_buffer.append(record)
                current_global_count += 1
        except Exception as e:
            logger.error(f"Error processing file {file_path.name}: {e}")

    if new_records_buffer:
        add_records_safely(client, dataset_name, workspace, new_records_buffer)
    else:
        logger.info("No new records to upload.")





def extract_annotations(row):
    """
    Safely extracts annotations, handling both grouped (dict) and flat (list/str) 
    response structures across multiple Response objects per user.
    """
    user_id_to_data = {}
    responses = row.get("responses")
    sentence = row.get("sentence_field", "")
    
    if not isinstance(responses, list):
        return []
        
    for resp in responses:
        if not isinstance(resp, dict):
            continue
            
        # 1. Clean Status (handles ResponseStatus.submitted enum)
        status_raw = resp.get("status", "")
        status = str(status_raw).split(".")[-1].lower()
        if status != "submitted":
            continue
            
        u_id = resp.get("user_id", "unknown")
        username = resp.get("username", "unknown")
        
        if u_id not in user_id_to_data:
            user_id_to_data[u_id] = {
                "username": username,
                "entities": [], 
                "suggestion": ""
            }
            
        # 2. Get the value (resp.values from the DataFrame)
        val = resp.get("values")
        
        # 3. Identify content type and MERGE (don't overwrite)
        if isinstance(val, list):
            # It's a list of spans
            user_id_to_data[u_id]["entities"].extend(val)
        elif isinstance(val, str):
            # It's a text suggestion (e.g. "ok")
            user_id_to_data[u_id]["suggestion"] = val
        elif isinstance(val, dict):
            # It's the standard grouped v2 format
            if "entities" in val:
                ent = val["entities"]
                spans = ent.get("value", ent) if isinstance(ent, dict) else ent
                if isinstance(spans, list):
                    user_id_to_data[u_id]["entities"].extend(spans)
            if "label_suggestion" in val:
                sug = val["label_suggestion"]
                sug_text = sug.get("value", sug) if isinstance(sug, dict) else sug
                user_id_to_data[u_id]["suggestion"] = str(sug_text)
        
    # 4. Format the final output
    results = []
    for u_id, data in user_id_to_data.items():
        entity_summaries = []
        
        # Deduplicate and sort spans by start position for readability
        unique_spans = []
        seen_spans = set()
        for s in data["entities"]:
            if not isinstance(s, dict): continue
            span_key = (s.get('start'), s.get('end'), s.get('label'))
            if span_key not in seen_spans:
                unique_spans.append(s)
                seen_spans.add(span_key)
        
        sorted_spans = sorted(unique_spans, key=lambda x: x.get('start', 0))

        for s in sorted_spans:
            try:
                start, end = s["start"], s["end"]
                label = s["label"]
                # Safely extract text from sentence
                text = sentence[start:end] if (sentence and 0 <= start < end <= len(sentence)) else f"[{start}:{end}]"
                entity_summaries.append(f"{text} [{label}]")
            except (KeyError, TypeError):
                continue
        
        details = ", ".join(entity_summaries)
        
        # Include all suggestions/notes
        sug_text = str(data["suggestion"]).strip()
        if sug_text:
            suffix = f" (Note: {sug_text})"
            details = f"{details}{suffix}" if details else sug_text
            
        # Include any user who submitted, even if they found 0 entities
        # This allows for "Negative Agreement" analysis
        raw_spans = [(s['start'], s['end'], s['label']) for s in sorted_spans]
        
        results.append({
            "annotator": data["username"],
            "user_id": u_id,
            "entities_count": len(entity_summaries),
            "details": details if details else "None (Negative Annotation)",
            "spans": raw_spans
        })
        
    return results

def prepare_iaa_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Groups annotations from the dataframe for Inter-Annotator Agreement analysis.
    
    Returns:
        A dictionary containing:
        - 'ready': List of dicts {id, text, annotations: {user: [(s,e,l), ...]}}
        - 'missing_teammate': List of sentence texts with only 1 annotator
        - 'unannotated': List of sentence texts with 0 annotators
    """
    ready = []
    missing_teammate = []
    unannotated = []
    
    for idx, row in df.iterrows():
        extracted = extract_annotations(row)
        count = len(extracted)
        
        if count >= 2:
            ready.append({
                "sentence_id": row.get("id", idx),
                "text": row["sentence_field"],
                "annotations": {ann["annotator"]: ann["spans"] for ann in extracted}
            })
        elif count == 1:
            missing_teammate.append(row["sentence_field"])
        else:
            unannotated.append(row["sentence_field"])
            
    return {
        "ready": ready,
        "missing_teammate": missing_teammate,
        "unannotated": unannotated
    }

def calculate_ner_iaa(iaa_ready: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Calculates comprehensive Inter-Annotator Agreement metrics:
    - Sentence-level Match %: Perfect identity between sets of annotations.
    - Span-level F1-Score: Traditional NER agreement on triplets (start, end, label).
    
    Args:
        iaa_ready: List of dictionaries from prepare_iaa_data["ready"]
        
    Returns:
        pd.DataFrame: A report comparing all annotator pairs.
    """
    from collections import defaultdict
    
    pairs = defaultdict(lambda: {
        "tp": 0, "fp": 0, "fn": 0, 
        "sentence_matches": 0, "total_sentences": 0
    })
    
    for item in iaa_ready:
        usernames = list(item["annotations"].keys())
        
        # Compare all unique pairs of annotators for this sentence
        for i in range(len(usernames)):
            for j in range(i + 1, len(usernames)):
                u1, u2 = usernames[i], usernames[j]
                pair_key = tuple(sorted([u1, u2]))
                
                set1 = set(item["annotations"][u1])
                set2 = set(item["annotations"][u2])
                
                # 1. Sentence-level Strict Match
                pairs[pair_key]["total_sentences"] += 1
                if set1 == set2:
                    pairs[pair_key]["sentence_matches"] += 1
                
                # 2. Span-level counts (standard F1 logic)
                tp = len(set1.intersection(set2))
                fp = len(set1 - set2)
                fn = len(set2 - set1)
                
                pairs[pair_key]["tp"] += tp
                pairs[pair_key]["fp"] += fp
                pairs[pair_key]["fn"] += fn

    results = []
    for (u1, u2), c in pairs.items():
        # F1 Calculation
        precision = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) > 0 else 0
        recall = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Match rate
        match_rate = c["sentence_matches"] / c["total_sentences"]
        
        results.append({
            "Annotator Pair": f"{u1} ↔ {u2}",
            "Sentences Compared": c["total_sentences"],
            "Perfect Matches": c["sentence_matches"],
            "Sentence Match %": f"{match_rate:.1%}",
            "Span F1-Score": round(f1, 4)
        })
    
    return pd.DataFrame(results)

def calculate_label_iaa(iaa_ready: List[Dict[str, Any]], annotators: List[str] = None) -> pd.DataFrame:
    """
    Provides a per-label breakdown of agreement metrics (F1-score).
    
    Args:
        iaa_ready: List of dictionaries from prepare_iaa_data["ready"]
        annotators: Optional list of two usernames [A, B] to compare. 
                    If None, the first pair found in the data is used.
        
    Returns:
        pd.DataFrame: Report with metrics for each entity type.
    """
    from collections import defaultdict
    
    label_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    
    # Identify the pair to compare
    if annotators is None:
        # Find the first pair in the dataset
        for item in iaa_ready:
            usernames = list(item["annotations"].keys())
            if len(usernames) >= 2:
                annotators = usernames[:2]
                break
    
    if not annotators or len(annotators) < 2:
        return pd.DataFrame()
        
    u1, u2 = annotators[0], annotators[1]
    
    for item in iaa_ready:
        annos = item["annotations"]
        if u1 not in annos or u2 not in annos:
            continue
            
        set1 = set(annos[u1])
        set2 = set(annos[u2])
        
        # Get all unique labels involved in this comparison
        all_labels = set([s[2] for s in set1] + [s[2] for s in set2])
        
        for label in all_labels:
            l_set1 = set([s for s in set1 if s[2] == label])
            l_set2 = set([s for s in set2 if s[2] == label])
                
            tp = len(l_set1.intersection(l_set2))
            fp = len(l_set1 - l_set2)
            fn = len(l_set2 - l_set1)
            
            label_stats[label]["tp"] += tp
            label_stats[label]["fp"] += fp
            label_stats[label]["fn"] += fn

    results = []
    total_tp = total_fp = total_fn = 0
    
    for label, c in label_stats.items():
        precision = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) > 0 else 0
        recall = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            "Label": label,
            "Both": c["tp"],
            f"Only in A ({u1})": c["fp"],
            f"Only in B ({u2})": c["fn"],
            "F1-Score": round(f1, 4)
        })
        
        total_tp += c["tp"]
        total_fp += c["fp"]
        total_fn += c["fn"]
    
    # Create DataFrame and sort by F1
    df_results = pd.DataFrame(results).sort_values("F1-Score", ascending=False)
    
    # Calculate Overall (Micro-average)
    overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * (overall_p * overall_r) / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0
    
    overall_row = pd.DataFrame([{
        "Label": "OVERALL",
        "Both": total_tp,
        f"Only in A ({u1})": total_fp,
        f"Only in B ({u2})": total_fn,
        "F1-Score": round(overall_f1, 4)
    }])
    
    return pd.concat([df_results, overall_row], ignore_index=True)

def analyze_iaa_discrepancies(iaa_ready: List[Dict[str, Any]], annotators: List[str] = None) -> pd.DataFrame:
    """
    Categorizes disagreements into Label Mismatches, Boundary Disagreements, and Misses.
    
    Returns:
        pd.DataFrame: A summary of agreement types.
    """
    if annotators is None or len(annotators) < 2:
        # Fallback to first two annotators found
        for item in iaa_ready:
            if len(item["annotations"]) >= 2:
                annotators = list(item["annotations"].keys())[:2]
                break
    
    if not annotators: return pd.DataFrame()
    u1, u2 = annotators[0], annotators[1]
    
    stats = {
        "Exact Matches": 0,
        "Label Mismatches": 0,       # Same span, different label
        "Boundary Disagreements": 0, # Overlapping spans
        "Total Misses (A only)": 0,   # No overlap found in B
        "Total Misses (B only)": 0    # No overlap found in A
    }
    total_char_diff = 0
    
    for item in iaa_ready:
        annos = item["annotations"]
        if u1 not in annos or u2 not in annos: continue
        
        spans1 = annos[u1] 
        spans2 = annos[u2]
        
        matched_in_2 = set()
        
        for s1 in spans1:
            start1, end1, label1 = s1
            found_match = False
            
            for i, s2 in enumerate(spans2):
                start2, end2, label2 = s2
                overlap = max(0, min(end1, end2) - max(start1, start2))
                
                if overlap > 0:
                    found_match = True
                    matched_in_2.add(i)
                    
                    if start1 == start2 and end1 == end2:
                        if label1 == label2:
                            stats["Exact Matches"] += 1
                        else:
                            stats["Label Mismatches"] += 1
                    else:
                        stats["Boundary Disagreements"] += 1
                        # Calculate total character shift at both boundaries
                        total_char_diff += abs(start1 - start2) + abs(end1 - end2)
                    break
            
            if not found_match:
                stats["Total Misses (A only)"] += 1
                
        for i in range(len(spans2)):
            if i not in matched_in_2:
                stats["Total Misses (B only)"] += 1
                
    df_stats = pd.DataFrame([stats]).T.rename(columns={0: "Count"})
    
    # Add average diff if there are disagreements
    if stats["Boundary Disagreements"] > 0:
        avg = round(total_char_diff / stats["Boundary Disagreements"], 2)
        avg_row = pd.DataFrame({"Count": [avg]}, index=["Boundary Char Diff (Avg)"])
        df_stats = pd.concat([df_stats, avg_row])

    return df_stats

def generate_adjudication_report_df(iaa_ready: List[Dict[str, Any]], annotators: List[str] = None) -> pd.DataFrame:
    """
    Generates a detailed DataFrame for adjudication, highlighting discrepancies between two annotators.
    """
    if annotators is None or len(annotators) < 2:
        # Fallback to first two annotators found in the first record that has >=2
        for item in iaa_ready:
            if len(item["annotations"]) >= 2:
                annotators = list(item["annotations"].keys())[:2]
                break
    
    if not annotators: return pd.DataFrame()
    u1, u2 = annotators[0], annotators[1]
    
    report_rows = []
    
    for item in iaa_ready:
        text = item["text"]
        annos = item["annotations"]
        if u1 not in annos or u2 not in annos: continue
        
        spans1 = sorted(annos[u1], key=lambda x: x[0])
        spans2 = sorted(annos[u2], key=lambda x: x[0])
        
        # Helper to format spans for display
        def format_spans(spans, text_content):
            lines = []
            for s in spans:
                start, end, label = s
                # Safely slice text
                snippet = text_content[start:end] if (0 <= start < end <= len(text_content)) else "???"
                lines.append(f"{snippet} [{label}] ({start}:{end})")
            return "\n".join(lines)
        
        disp1 = format_spans(spans1, text)
        disp2 = format_spans(spans2, text)
        
        # Identify discrepancies
        set1 = set(spans1)
        set2 = set(spans2)
        
        discreps = []
        status = "Match"
        
        if set1 != set2:
            matched_indices_in_2 = set()
            
            for s1 in spans1:
                start1, end1, label1 = s1
                found_overlap = False
                snippet1 = text[start1:end1] if (0 <= start1 < end1 <= len(text)) else "???"
                
                for i, s2 in enumerate(spans2):
                    start2, end2, label2 = s2
                    overlap = max(0, min(end1, end2) - max(start1, start2))
                    
                    if overlap > 0:
                        found_overlap = True
                        matched_indices_in_2.add(i)
                        
                        if start1 == start2 and end1 == end2:
                            if label1 != label2:
                                discreps.append(f"Label mismatch for '{snippet1}': {label1} ({u1}) vs {label2} ({u2})")
                                if status == "Match": status = "Label Mismatch"
                                elif status != "Label Mismatch": status = "Multiple Issues"
                        else:
                            snippet2 = text[start2:end2] if (0 <= start2 < end2 <= len(text)) else "???"
                            discreps.append(f"Boundary mismatch: '{snippet1}' ({u1}) vs '{snippet2}' ({u2})")
                            if status == "Match": status = "Boundary Mismatch"
                            elif status != "Boundary Mismatch": status = "Multiple Issues"
                        break # Found the primary overlap
                
                if not found_overlap:
                    discreps.append(f"Missed by {u2}: '{snippet1}' [{label1}] at {start1}:{end1}")
                    if status == "Match": status = f"Miss by {u2}"
                    elif status != f"Miss by {u2}": status = "Multiple Issues"
            
            for i, s2 in enumerate(spans2):
                if i not in matched_indices_in_2:
                    start2, end2, label2 = s2
                    snippet2 = text[start2:end2] if (0 <= start2 < end2 <= len(text)) else "???"
                    discreps.append(f"Missed by {u1}: '{snippet2}' [{label2}] at {start2}:{end2}")
                    if status == "Match": status = f"Miss by {u1}"
                    elif status not in [f"Miss by {u1}", "Multiple Issues"]: status = "Multiple Issues"

        report_rows.append({
            "sentence_id": item["sentence_id"],
            "text": text,
            f"{u1}_annotations": disp1 if disp1 else "(None)",
            f"{u2}_annotations": disp2 if disp2 else "(None)",
            "discrepancies": "\n".join(discreps) if discreps else "None",
            "status": status
        })
        
    return pd.DataFrame(report_rows)

def highlight_entities_html(text: str, spans: List[tuple], color_map: Dict[str, str] = None) -> str:
    """
    Returns an HTML string with entities highlighted.
    spans: List of (start, end, label)
    """
    if not color_map:
        color_map = {
            'ARTEFACT': '#e8f4f8', 'PERIOD': '#fff3cd', 'LOCATION': '#d4edda',
            'CONTEXT': '#d1ecf1', 'CONTEXT_ID': '#f8d7da', 'MATERIAL': '#e2e3e5',
            'SPECIES': '#f3e5f5', 'FEATURE': '#fff9c4', 'PERSON': '#c8e6c9', 'MISC': '#eeeeee'
        }
    
    # Sort spans by start, handle overlaps (simple strategy: take first)
    sorted_spans = sorted(spans, key=lambda x: x[0])
    
    html = ""
    last_idx = 0
    
    for start, end, label in sorted_spans:
        if start < last_idx: continue # Skip overlapping for simple viz
        
        # Text before entity
        html += text[last_idx:start]
        
        # Entity with background color and label
        color = color_map.get(label, '#ffffff')
        entity_text = text[start:end]
        html += (
            f'<span style="background-color: {color}; border-radius: 4px; padding: 2px 4px; margin: 0 2px; border: 1px solid #ccc;" '
            f'title="{label}">'
            f'{entity_text} <b style="font-size: 0.8em; opacity: 0.7;">({label})</b>'
            f'</span>'
        )
        last_idx = end
        
    html += text[last_idx:]
    return f'<div style="line-height: 2.0; font-family: sans-serif; font-size: 1.1em;">{html}</div>'

def highlight_diff_entities_html(text: str, spans_a: list, spans_b: list, name_a: str = "A", name_b: str = "B") -> str:
    """
    Renders an HTML view where ONLY the differences between spans_a and spans_b are highlighted.
    Perfect matches are ignored.
    """
    set_a = set(tuple(s) for s in spans_a if len(s)==3)
    set_b = set(tuple(s) for s in spans_b if len(s)==3)
    matches = set_a & set_b
    
    diff_a = set_a - matches
    diff_b = set_b - matches
    
    char_states = [{'a': set(), 'b': set()} for _ in range(len(text))]
    
    for s, e, l in diff_a:
        for i in range(s, e):
            if 0 <= i < len(text):
                char_states[i]['a'].add(l)
                
    for s, e, l in diff_b:
        for i in range(s, e):
            if 0 <= i < len(text):
                char_states[i]['b'].add(l)
                
    chunks = []
    current_state = None
    current_start = 0
    
    for i in range(len(text)):
        a_labels = tuple(sorted(char_states[i]['a']))
        b_labels = tuple(sorted(char_states[i]['b']))
        state = (a_labels, b_labels)
        
        if state != current_state:
            if current_state is not None:
                chunks.append((current_start, i, current_state))
            current_state = state
            current_start = i
            
    if current_state is not None:
        chunks.append((current_start, len(text), current_state))
        
    html = ""
    for start, end, state in chunks:
        a_labels, b_labels = state
        chunk_text = text[start:end]
        
        if not a_labels and not b_labels:
            html += chunk_text
        else:
            bg_color = "#ffeeee" # default light red
            if a_labels and not b_labels:
                bg_color = "#e6f2ff" # light blue
            elif b_labels and not a_labels:
                bg_color = "#e6ffe6" # light green
            elif a_labels and b_labels:
                bg_color = "#fff0b3" # yellow for conflict
                
            title = ""
            if a_labels: title += f"{name_a}: {', '.join(a_labels)} "
            if b_labels: title += f"{name_b}: {', '.join(b_labels)}"
            
            labels_html = ""
            if a_labels:
                labels_html += f"<span style='color: #0055aa; font-weight: bold; font-size: 0.7em; vertical-align: super;'>[{name_a}: {','.join(a_labels)}]</span> "
            if b_labels:
                labels_html += f"<span style='color: #008800; font-weight: bold; font-size: 0.7em; vertical-align: sub;'>[{name_b}: {','.join(b_labels)}]</span> "
                
            html += f"<span style='background-color: {bg_color}; border: 1px dashed #ccc; padding: 2px 4px; border-radius: 4px;' title='{title}'>{chunk_text}{labels_html}</span>"
            
    return f"<div style='line-height: 2.5; font-family: sans-serif; font-size: 1.1em;'>{html}</div>"

def render_interactive_adjudication_tool(iaa_ready: list, discrepancies_only: pd.DataFrame, annotator_a: str, annotator_b: str):
    """
    Renders the interactive widget UI for reviewing adjudication discrepancies.
    Should be called at the end of a Jupyter Notebook cell.
    """
    import ipywidgets as widgets
    from IPython.display import display, HTML
    
    # Filter iaa_ready to match the discrepancies-only index mapping
    discrepancy_ids = set(discrepancies_only["sentence_id"])
    iaa_discrepancies = [item for item in iaa_ready if item["sentence_id"] in discrepancy_ids]

    if not iaa_discrepancies:
        display(HTML("<p>No discrepancies found to review.</p>"))
        return

    # Create widgets
    record_slider = widgets.IntSlider(
        value=0, min=0, max=len(iaa_discrepancies)-1, 
        description='Sentence:', 
        layout=widgets.Layout(width='40%')
    )

    prev_button = widgets.Button(description='Previous', icon='chevron-left')
    next_button = widgets.Button(description='Next', icon='chevron-right')

    annotator_toggle = widgets.Dropdown(
        options=[
            (annotator_a, annotator_a), 
            (annotator_b, annotator_b), 
            ('Compare (Stacked)', 'compare'), 
            ('Diff (Merge)', 'diff')
        ],
        value='diff',
        description='View:',
    )

    # Use HTML widget for stability - clears previous content automatically on .value update
    html_viewer = widgets.HTML()

    def update_view(change):
        if len(iaa_discrepancies) == 0: 
            html_viewer.value = "No discrepancies found."
            return
            
        idx = record_slider.value
        user = annotator_toggle.value
        item = iaa_discrepancies[idx]
        
        text = item["text"]
        
        # Get discrepancy info from our report DF
        report_row = discrepancies_only[discrepancies_only["sentence_id"] == item["sentence_id"]].iloc[0]
        
        html_out = f"<h3>Sentence ID: {item['sentence_id']} ({idx + 1} / {len(iaa_discrepancies)})</h3>"
        
        if user == 'compare':
            spans_a = item["annotations"].get(annotator_a, [])
            spans_b = item["annotations"].get(annotator_b, [])
            
            html_out += f"<div style='border: 1px solid #ccc; padding: 10px; margin-bottom: 10px;'>"
            html_out += f"<h4 style='margin-top: 0;'>{annotator_a}</h4>"
            if annotator_a not in item["annotations"]:
                html_out += f"<p style='color: red;'><b>Warning:</b> User '{annotator_a}' not found.</p>"
            html_out += highlight_entities_html(text, spans_a)
            html_out += "</div>"
            
            html_out += f"<div style='border: 1px solid #ccc; padding: 10px; margin-bottom: 10px;'>"
            html_out += f"<h4 style='margin-top: 0;'>{annotator_b}</h4>"
            if annotator_b not in item["annotations"]:
                html_out += f"<p style='color: red;'><b>Warning:</b> User '{annotator_b}' not found.</p>"
            html_out += highlight_entities_html(text, spans_b)
            html_out += "</div>"
            
        elif user == 'diff':
            spans_a = item["annotations"].get(annotator_a, [])
            spans_b = item["annotations"].get(annotator_b, [])
            html_out += f"<div style='border: 1px solid #ccc; padding: 10px; margin-bottom: 10px;'>"
            html_out += f"<h4 style='margin-top: 0;'>Combined Diff (Showing Differences Only)</h4>"
            html_out += highlight_diff_entities_html(text, spans_a, spans_b, annotator_a, annotator_b)
            html_out += "</div>"
            
        else:
            spans = item["annotations"].get(user, [])
            html_out += f"<p><b>Showing annotations for:</b> {user}</p>"
            
            if user not in item["annotations"]:
                html_out += f"<p style='color: red;'><b>Warning:</b> User '{user}' not found in record. Available keys: {list(item['annotations'].keys())}</p>"
            
            html_out += highlight_entities_html(text, spans)
            
        html_out += "<hr>"
        html_out += "<h4>Discrepancy Details (Ref):</h4>"
        html_out += f"<pre style='background: #f8f9fa; padding: 10px; border: 1px solid #ddd;'>{report_row['discrepancies']}</pre>"
        
        # Simple assignment ensures NO duplication
        html_viewer.value = html_out

    def on_prev_clicked(b):
        if record_slider.value > 0:
            record_slider.value -= 1

    def on_next_clicked(b):
        if record_slider.value < record_slider.max:
            record_slider.value += 1

    record_slider.observe(update_view, names='value')
    annotator_toggle.observe(update_view, names='value')
    prev_button.on_click(on_prev_clicked)
    next_button.on_click(on_next_clicked)

    # Initial display
    nav_box = widgets.HBox([prev_button, record_slider, next_button])
    ui = widgets.VBox([nav_box, annotator_toggle])
    display(ui)
    display(html_viewer)
    update_view(None)
