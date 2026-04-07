# Archaeo-NER-Greek

Welcome! This project is dedicated to **Named Entity Recognition (NER)** for Greek archaeological texts. It provides tools for data preparation, annotation management via Argilla, and Inter-Annotator Agreement (IAA) analysis.

---

## Part 1: Setting Up Your Environment

Before you can run any scripts, you need to install four essential tools.

### 1. VS Code or Antigravity (The Editor)
You can use **VS Code** as your primary editor, or leverage **Antigravity**, a powerful AI coding assistant that helps you navigate the codebase, run scripts, and manage your data workflows through natural language commands.

#### Connecting your Google Account
To access professional features and specialized AI models:
- **In Antigravity**: Click on the user profile icon (usually in the top right or bottom left corner depending on your interface) and select **"Sign in with Google"**. Ensure you use the account associated with your Pro subscription.
- **In VS Code**: If you are using the official Google extensions, look for the account icon in the Activity Bar (left side). Click it and select **"Sign in to sync settings"** or **"Sign in with Google"** to link your environment with your account.

### 2. Python (The Engine)
- This project requires **Python 3.12** or higher.
- **Linux**: Usually comes pre-installed. Verify with `python3 --version`.
- **Windows**: Download from [python.org](https://www.python.org/downloads/). **CRITICAL**: Check **"Add Python to PATH"** during installation.

### 3. Git
- **Windows**: Download "Git for Windows" from [git-scm.com](https://git-scm.com/).
- **Linux**: Install via your terminal (e.g., `sudo apt install git`).

### 4. uv (The Manager)
`uv` is a modern tool that automatically manages your virtual environment and dependencies.
- **Windows (PowerShell)**:
  ```powershell
  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
- **Linux/macOS**:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

---

## Part 2: Getting the Code

1. Open a terminal and navigate to your projects folder.
2. Clone the repository:
   ```bash
   git clone https://github.com/prokopidis/archaeo-ner-greek.git
   cd archaeo-ner-greek
   ```
3. Initialize the environment:
   ```bash
   uv sync
   ```

---

## Part 3: Configuration (.env file)

The project uses [Argilla](https://argilla.io/) for data annotation. You need to provide your Argilla credentials in a `.env` file.

1. In the project root, create a file named `.env`.
2. Copy the content from `.env.example` and fill in your details:
   ```env
   ARGILLA_API_URL=https://your-argilla-instance.com
   ARGILLA_API_KEY=your-api-key
   ```

---

## Part 4: Project Structure

- `archaeo_ner_greek/`: Core Python package containing utilities for Argilla integration, logging, and data processing.
- `notebooks/`: Jupyter notebooks for various tasks:
  - `iaa_from_argilla_data.ipynb`: Calculate Inter-Annotator Agreement.
- `data/`: Contains sample texts and annotation guidelines.
- `models/`: Directory for storing trained NER models.

---

## Part 5: Usage

Most tasks are currently performed via Jupyter notebooks. To start working:

1. Activate the environment:
   ```bash
   source .venv/bin/activate  # On Linux/macOS
   .venv\Scripts\activate     # On Windows
   ```
2. Use **VS Code** or **Antigravity** to work with the project.

### Working with Jupyter Notebooks

Jupyter notebooks (files ending in `.ipynb`) allow you to run code in "blocks" or "cells". You can use `notebooks/iaa_from_argilla_data.ipynb` as a starting point.

#### 1. Loading the Kernel
The "Kernel" is the engine that runs your code. 
- When you open a notebook, look at the top right corner.
- If it says **"Select Kernel"**, click it and choose the Python environment created by `uv` (usually labeled as `.venv` or `Python 3.12.x`).

#### 2. Running Code Blocks
- **Run a single cell**: Click the **Play icon** (▶) next to the cell, or press `Shift + Enter`.
- **Run all cells**: Click **"Run All"** in the top toolbar to execute the entire notebook from start to finish.

#### 3. Common Shortcuts
- `Shift + Enter`: Run the current cell and move to the next.
- `Ctrl + Enter`: Run the current cell and stay on it.


#### 4. Managing the Kernel
If the code gets stuck or you want to start fresh:
- **Interrupt** (⏹): Stops the code that is currently running.
- **Restart**: Clears all memory and variables. You will need to run the cells again from the top.
- **Clear All Outputs**: Removes the results shown below the cells to make the notebook cleaner. This does not delete your code.

---
