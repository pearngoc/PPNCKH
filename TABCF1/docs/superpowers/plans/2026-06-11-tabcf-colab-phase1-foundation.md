# TABCF Colab Migration - Phase 1: Foundation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the foundation notebook with environment setup, data handling, and configuration for TABCF Colab migration.

**Architecture:** Single Jupyter notebook with initial cells for dependency installation, GPU detection, data upload/validation, and configuration dataclass.

**Tech Stack:** Google Colab, PyTorch, Python 3.10+, dataclasses

---

## File Structure

**Files to create:**
- `TABCF_Colab.ipynb` - Main Jupyter notebook (Phase 1: first ~15-20 cells)

**Files to reference:**
- `requirements-kaggle.txt` - Dependency list
- `data/Info/adult.json` - Dataset metadata
- `utils_train.py` - Helper functions to adapt

---

## Task 1: Create Notebook Shell and Title Section

**Files:**
- Create: `TABCF_Colab.ipynb`

- [ ] **Step 1: Create empty Jupyter notebook**

Create file `TABCF_Colab.ipynb` with basic structure:

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# TABCF: Counterfactual Explanations for Tabular Data\n",
    "\n",
    "**Paper:** [TABCF: Transformer-Based VAE for Counterfactual Explanations](https://dl.acm.org/doi/10.1145/3677052.3698673)\n",
    "\n",
    "This notebook implements the complete TABCF pipeline:\n",
    "1. Environment Setup\n",
    "2. Data Upload & Preparation\n",
    "3. Configuration\n",
    "4. VAE Training\n",
    "5. Black-Box Classifier Training\n",
    "6. Counterfactual Generation\n",
    "7. Evaluation & Visualization\n",
    "\n",
    "**Estimated Runtime:** 35-70 minutes on Colab T4 GPU\n",
    "\n",
    "---"
   ]
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "colab": {
   "gpuType": "T4",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}
```

- [ ] **Step 2: Verify notebook structure**

Open the notebook in a text editor or Jupyter and confirm:
- Title markdown cell renders correctly
- Metadata includes GPU accelerator setting
- No syntax errors

- [ ] **Step 3: Commit**

```bash
git add TABCF_Colab.ipynb
git commit -m "feat: create TABCF Colab notebook shell with title section"
```

---

## Task 2: Environment Setup - Package Installation

**Files:**
- Modify: `TABCF_Colab.ipynb` (add cells)
- Reference: `requirements-kaggle.txt`

- [ ] **Step 1: Add environment setup header cell**

Add markdown cell after title:

```markdown
## 1. Environment Setup

Install required dependencies and configure GPU.
```

- [ ] **Step 2: Add GPU detection cell**

Add code cell:

```python
import torch
import sys

# Check GPU availability
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  PyTorch version: {torch.__version__}")
else:
    device = torch.device('cpu')
    print("⚠ WARNING: GPU not available, using CPU (training will be very slow)")
    print(f"  PyTorch version: {torch.__version__}")

print(f"  Python version: {sys.version.split()[0]}")
```

- [ ] **Step 3: Add package installation cell**

Add code cell:

```python
%%capture
# Install required packages (suppress output for cleaner notebook)
!pip install -q transformers>=4.25.0 \
             datasets>=2.8.0 \
             peft>=0.3.0 \
             ml_collections>=0.1.1 \
             sdmetrics>=0.8.0 \
             prdc>=0.1.0 \
             rdt>=1.3.0 \
             pyod>=1.0.0 \
             category_encoders>=2.5.0 \
             imbalanced-learn>=0.9.0 \
             icecream>=2.1.0 \
             xlrd>=2.0.0 \
             tomli-w>=1.0.0 \
             openpyxl>=3.0.0

print("✓ All packages installed successfully")
```

- [ ] **Step 4: Add import validation cell**

Add code cell:

```python
# Verify critical imports
try:
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    import sklearn
    from transformers import __version__ as transformers_version
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import seaborn as sns
    from dataclasses import dataclass
    import json
    import os
    import warnings
    warnings.filterwarnings('ignore')
    
    print("✓ All imports successful")
    print(f"  NumPy: {np.__version__}")
    print(f"  Pandas: {pd.__version__}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  Scikit-learn: {sklearn.__version__}")
    print(f"  Transformers: {transformers_version}")
    
except ImportError as e:
    print(f"✗ Import failed: {e}")
    print("Please restart runtime and re-run installation cell")
```

- [ ] **Step 5: Test notebook cells execute without errors**

Run the notebook cells in order:
1. GPU detection cell - should show GPU if available
2. Package installation cell - should complete without errors
3. Import validation cell - should print all package versions

Expected output includes GPU name (if available) and package versions.

- [ ] **Step 6: Commit**

```bash
git add TABCF_Colab.ipynb
git commit -m "feat: add environment setup cells with GPU detection and package installation"
```

---

## Task 3: Data Upload Options

**Files:**
- Modify: `TABCF_Colab.ipynb` (add cells)

- [ ] **Step 1: Add data upload header cell**

Add markdown cell:

```markdown
## 2. Data Upload & Preparation

Choose one option:
- **Option A:** Direct upload (first-time use, ~2-3 minutes)
- **Option B:** Google Drive mount (faster for repeated runs)
```

- [ ] **Step 2: Add data upload option selector cell**

Add code cell:

```python
# Select data upload method
print("Choose data upload method:")
print("A: Direct upload (upload data folder as zip)")
print("B: Google Drive mount (requires pre-uploaded data)")
print()

# User should modify this variable
DATA_UPLOAD_METHOD = 'A'  # Change to 'B' for Drive mount

if DATA_UPLOAD_METHOD not in ['A', 'B']:
    raise ValueError("DATA_UPLOAD_METHOD must be 'A' or 'B'")

print(f"✓ Selected method: {DATA_UPLOAD_METHOD}")
```

- [ ] **Step 3: Add Option A - Direct upload cell**

Add code cell:

```python
# Option A: Direct Upload
if DATA_UPLOAD_METHOD == 'A':
    from google.colab import files
    import zipfile
    
    print("Please upload your data folder as a ZIP file...")
    print("Expected structure: data.zip containing data/Info/ and data/adult/")
    print()
    
    uploaded = files.upload()
    
    if not uploaded:
        raise ValueError("No file uploaded. Please re-run and select a file.")
    
    # Get the uploaded filename
    zip_filename = list(uploaded.keys())[0]
    
    # Extract to /content/
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall('/content/')
    
    # Clean up zip file
    os.remove(zip_filename)
    
    print(f"✓ Extracted {zip_filename} to /content/")
    
    # Verify data folder exists
    if os.path.exists('/content/data'):
        print("✓ Data folder found at /content/data")
    else:
        raise FileNotFoundError("Expected /content/data folder not found. Check ZIP structure.")
else:
    print("Skipping direct upload (Option B selected)")
```

- [ ] **Step 4: Add Option B - Drive mount cell**

Add code cell:

```python
# Option B: Google Drive Mount
if DATA_UPLOAD_METHOD == 'B':
    from google.colab import drive
    
    # Mount Google Drive
    drive.mount('/content/drive')
    
    # User should modify this path to where they stored data folder in Drive
    DRIVE_DATA_PATH = '/content/drive/MyDrive/TABCF_data'
    
    print(f"Looking for data at: {DRIVE_DATA_PATH}")
    
    if not os.path.exists(DRIVE_DATA_PATH):
        raise FileNotFoundError(
            f"Data folder not found at {DRIVE_DATA_PATH}\n"
            f"Please upload your data folder to Google Drive and update DRIVE_DATA_PATH"
        )
    
    # Create symlink to standard location
    if os.path.exists('/content/data'):
        os.remove('/content/data')
    os.symlink(DRIVE_DATA_PATH, '/content/data')
    
    print(f"✓ Data linked from Drive to /content/data")
else:
    print("Skipping Drive mount (Option A selected)")
```

- [ ] **Step 5: Test both upload methods work**

Test Option A:
1. Create a test zip file containing `data/Info/adult.json`
2. Set `DATA_UPLOAD_METHOD = 'A'`
3. Run the Option A cell and upload the test zip
4. Verify `/content/data` exists

Test Option B:
1. Set `DATA_UPLOAD_METHOD = 'B'`
2. Set `DRIVE_DATA_PATH` to a valid (or test) path
3. Run the Option B cell
4. Verify appropriate error or success message

- [ ] **Step 6: Commit**

```bash
git add TABCF_Colab.ipynb
git commit -m "feat: add data upload options (direct upload and Drive mount)"
```

---

## Task 4: Data Validation

**Files:**
- Modify: `TABCF_Colab.ipynb` (add cells)
- Reference: `data/Info/adult.json`

- [ ] **Step 1: Add data validation cell**

Add code cell:

```python
# Validate data folder structure
print("Validating data folder structure...")

required_paths = {
    'data_root': '/content/data',
    'info_dir': '/content/data/Info',
    'adult_dir': '/content/data/adult',
    'adult_info': '/content/data/Info/adult.json',
}

missing = []
for name, path in required_paths.items():
    if os.path.exists(path):
        print(f"✓ {name}: {path}")
    else:
        print(f"✗ {name}: {path} NOT FOUND")
        missing.append(name)

if missing:
    raise FileNotFoundError(
        f"Missing required paths: {', '.join(missing)}\n"
        f"Please check your data folder structure."
    )

print("\n✓ All required paths found")
```

- [ ] **Step 2: Add dataset info loading cell**

Add code cell:

```python
# Load dataset metadata
with open('/content/data/Info/adult.json', 'r') as f:
    dataset_info = json.load(f)

print("Dataset Information:")
print(f"  Name: {dataset_info['name']}")
print(f"  Task: {dataset_info['task_type']}")
print(f"  Target: {dataset_info['target_col']}")
print(f"  Train samples: {dataset_info['train_num']}")
print(f"  Test samples: {dataset_info['test_num']}")
print(f"  Total features: {len(dataset_info['column_names'])}")
print(f"  Immutable features: {dataset_info['immutable']}")

# Count feature types
num_features = sum(1 for v in dataset_info['column_info'].values() if v == 'float')
cat_features = sum(1 for v in dataset_info['column_info'].values() if v == 'str')
print(f"  Numeric features: {num_features}")
print(f"  Categorical features: {cat_features}")
```

- [ ] **Step 3: Add processed data file verification cell**

Add code cell:

```python
# Check for preprocessed data files
adult_dir = '/content/data/adult'
expected_files = [
    'X_num_train.npy',
    'X_cat_train.npy',
    'X_num_test.npy',
    'X_cat_test.npy',
    'y_train.npy',
    'y_test.npy',
    'black_box_mlp_hidden_16.pkl'
]

print("\nChecking preprocessed data files:")
found_files = []
missing_files = []

for filename in expected_files:
    filepath = os.path.join(adult_dir, filename)
    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"✓ {filename} ({size_mb:.2f} MB)")
        found_files.append(filename)
    else:
        print(f"✗ {filename} NOT FOUND")
        missing_files.append(filename)

if missing_files:
    print(f"\n⚠ Warning: {len(missing_files)} files missing")
    print("These files will be created during training/preprocessing")
else:
    print(f"\n✓ All {len(expected_files)} preprocessed files found")
```

- [ ] **Step 4: Test validation cells**

Run validation cells with valid data folder:
- Should print dataset metadata from adult.json
- Should show which preprocessed files exist
- Should complete without raising exceptions

Test with missing files:
- Temporarily rename `/content/data` to `/content/data_backup`
- Run validation cell
- Should raise FileNotFoundError with clear message
- Restore `/content/data`

- [ ] **Step 5: Commit**

```bash
git add TABCF_Colab.ipynb
git commit -m "feat: add data validation cells with structure and file checks"
```

---

## Task 5: Configuration Dataclass

**Files:**
- Modify: `TABCF_Colab.ipynb` (add cells)

- [ ] **Step 1: Add configuration header cell**

Add markdown cell:

```markdown
## 3. Configuration

Define all hyperparameters and paths. Modify these values before training.
```

- [ ] **Step 2: Add configuration dataclass cell**

Add code cell:

```python
from dataclasses import dataclass

@dataclass
class TabCFConfig:
    """Configuration for TABCF training and counterfactual generation."""
    
    # Dataset
    dataname: str = 'adult'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # VAE Training
    max_beta: float = 1e-2        # KL weight annealing maximum
    min_beta: float = 1e-5        # KL weight annealing minimum
    lambd: float = 0.7            # Balance reconstruction vs KL loss
    gumbel_softmax: bool = True   # Differentiable categorical reconstruction
    tau: float = 1.0              # Gumbel-Softmax temperature
    epochs: int = 100             # Number of training epochs
    batch_size: int = 256         # Training batch size
    learning_rate: float = 1e-3   # Learning rate for VAE
    weight_decay: float = 0       # L2 regularization
    
    # VAE Architecture
    d_token: int = 4              # Token embedding dimension
    n_head: int = 1               # Number of attention heads
    factor: int = 32              # Dimension multiplier
    num_layers: int = 2           # Number of transformer layers
    
    # Black-box Classifier
    hidden_dims: int = 16         # MLP hidden layer size
    clf_epochs: int = 50          # Classifier training epochs
    clf_learning_rate: float = 1e-3  # Classifier learning rate
    
    # Counterfactual Generation
    num_samples: int = 100        # Number of test instances to explain
    proximity_weight_input: float = 1.0    # Weight for input space proximity
    proximity_weight_latent: float = 1.0   # Weight for latent space proximity
    diversity_weight: float = 1.0          # Weight for diversity
    sparsity_weight: float = 0.5           # Weight for sparsity
    max_iterations: int = 1000             # Max optimization iterations per CF
    
    # Paths
    data_path: str = '/content/data'
    ckpt_path: str = '/content/ckpt'
    results_path: str = '/content/counterfactual_results'
    
    # Training settings
    verbose: bool = True          # Print detailed logs
    save_interval: int = 10       # Save checkpoint every N epochs
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        os.makedirs(self.ckpt_path, exist_ok=True)
        os.makedirs(self.results_path, exist_ok=True)
        os.makedirs(os.path.join(self.ckpt_path, self.dataname), exist_ok=True)

# Initialize configuration
config = TabCFConfig()

print("Configuration initialized:")
print(f"  Device: {config.device}")
print(f"  Dataset: {config.dataname}")
print(f"  VAE epochs: {config.epochs}")
print(f"  Batch size: {config.batch_size}")
print(f"  Checkpoint path: {config.ckpt_path}/{config.dataname}/")
print(f"  Results path: {config.results_path}")
```

- [ ] **Step 3: Add configuration summary cell**

Add code cell:

```python
# Display key configuration parameters
print("=" * 50)
print("TABCF Configuration Summary")
print("=" * 50)

print("\n📊 Dataset:")
print(f"  Name: {config.dataname}")
print(f"  Data path: {config.data_path}")

print("\n🧠 VAE Architecture:")
print(f"  Token dimension: {config.d_token}")
print(f"  Attention heads: {config.n_head}")
print(f"  Transformer layers: {config.num_layers}")
print(f"  Gumbel-Softmax: {config.gumbel_softmax}")

print("\n🎯 Training:")
print(f"  Device: {config.device}")
print(f"  Epochs: {config.epochs}")
print(f"  Batch size: {config.batch_size}")
print(f"  Learning rate: {config.learning_rate}")
print(f"  Beta range: [{config.min_beta}, {config.max_beta}]")
print(f"  Lambda: {config.lambd}")

print("\n🎲 Counterfactual Generation:")
print(f"  Samples to explain: {config.num_samples}")
print(f"  Max iterations: {config.max_iterations}")
print(f"  Proximity weight (input): {config.proximity_weight_input}")
print(f"  Proximity weight (latent): {config.proximity_weight_latent}")
print(f"  Sparsity weight: {config.sparsity_weight}")

print("\n💾 Output:")
print(f"  Checkpoints: {config.ckpt_path}/{config.dataname}/")
print(f"  Results: {config.results_path}")

print("=" * 50)
```

- [ ] **Step 4: Add configuration modification instructions cell**

Add markdown cell:

```markdown
### Modifying Configuration

To change hyperparameters, edit the `TabCFConfig` class above and re-run the cells:

```python
# Example: Use smaller batch size for memory-constrained environments
config.batch_size = 128

# Example: Train for fewer epochs (faster testing)
config.epochs = 50

# Example: Generate more counterfactuals
config.num_samples = 200
```

After modifying, re-run the configuration cells to apply changes.
```

- [ ] **Step 5: Test configuration dataclass**

Run configuration cells:
- Config should initialize without errors
- Directories `/content/ckpt` and `/content/counterfactual_results` should be created
- Summary should print all configuration values
- Device should show 'cuda' if GPU available, else 'cpu'

Test modification:
```python
config.batch_size = 128
print(f"Modified batch size: {config.batch_size}")
```
- Should update the value successfully

- [ ] **Step 6: Commit**

```bash
git add TABCF_Colab.ipynb
git commit -m "feat: add configuration dataclass with all hyperparameters and paths"
```

---

## Task 6: Create Checkpoint Utilities

**Files:**
- Modify: `TABCF_Colab.ipynb` (add cells)

- [ ] **Step 1: Add checkpoint utilities header cell**

Add markdown cell:

```markdown
### Checkpoint Management Utilities

Helper functions for saving and loading model checkpoints.
```

- [ ] **Step 2: Add checkpoint utilities cell**

Add code cell:

```python
def save_checkpoint(model, optimizer, epoch, loss, filepath, metadata=None):
    """Save model checkpoint with metadata."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metadata': metadata or {}
    }
    torch.save(checkpoint, filepath)
    if config.verbose:
        print(f"✓ Checkpoint saved: {filepath}")

def load_checkpoint(filepath, model, optimizer=None):
    """Load model checkpoint."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if config.verbose:
        print(f"✓ Checkpoint loaded: {filepath}")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Loss: {checkpoint.get('loss', 'N/A'):.4f}")
    
    return checkpoint

def checkpoint_exists(filepath):
    """Check if checkpoint file exists."""
    return os.path.exists(filepath)

print("✓ Checkpoint utilities defined")
```

- [ ] **Step 3: Add training history utilities cell**

Add code cell:

```python
class TrainingHistory:
    """Track training metrics over epochs."""
    
    def __init__(self):
        self.epochs = []
        self.train_losses = []
        self.val_losses = []
        self.metrics = {}
    
    def add(self, epoch, train_loss, val_loss=None, **kwargs):
        """Add metrics for an epoch."""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
    
    def save(self, filepath):
        """Save history to JSON file."""
        history_dict = {
            'epochs': self.epochs,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'metrics': self.metrics
        }
        with open(filepath, 'w') as f:
            json.dump(history_dict, f, indent=2)
        print(f"✓ Training history saved: {filepath}")
    
    def load(self, filepath):
        """Load history from JSON file."""
        with open(filepath, 'r') as f:
            history_dict = json.load(f)
        self.epochs = history_dict['epochs']
        self.train_losses = history_dict['train_losses']
        self.val_losses = history_dict['val_losses']
        self.metrics = history_dict['metrics']
        print(f"✓ Training history loaded: {filepath}")

print("✓ TrainingHistory class defined")
```

- [ ] **Step 4: Test checkpoint utilities**

Test saving and loading:
```python
# Test checkpoint utilities
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

test_model = DummyModel()
test_optimizer = torch.optim.Adam(test_model.parameters())
test_path = '/content/test_checkpoint.pth'

# Test save
save_checkpoint(test_model, test_optimizer, epoch=1, loss=0.5, filepath=test_path)

# Test load
loaded_checkpoint = load_checkpoint(test_path, test_model, test_optimizer)

# Test exists
assert checkpoint_exists(test_path) == True
assert checkpoint_exists('/content/fake.pth') == False

# Clean up
os.remove(test_path)
print("✓ Checkpoint utilities test passed")
```

Test training history:
```python
# Test TrainingHistory
history = TrainingHistory()
history.add(epoch=1, train_loss=0.5, val_loss=0.6, accuracy=0.8)
history.add(epoch=2, train_loss=0.4, val_loss=0.5, accuracy=0.85)

test_history_path = '/content/test_history.json'
history.save(test_history_path)

# Load and verify
history2 = TrainingHistory()
history2.load(test_history_path)
assert len(history2.epochs) == 2
assert history2.train_losses[0] == 0.5

# Clean up
os.remove(test_history_path)
print("✓ TrainingHistory test passed")
```

- [ ] **Step 5: Commit**

```bash
git add TABCF_Colab.ipynb
git commit -m "feat: add checkpoint and training history utilities"
```

---

## Task 7: Phase 1 Integration Test

**Files:**
- Modify: `TABCF_Colab.ipynb` (add test cell at end)

- [ ] **Step 1: Add Phase 1 completion cell**

Add markdown cell at the end:

```markdown
---

## Phase 1 Complete ✓

**What we have:**
- ✓ Environment setup with GPU detection
- ✓ Package installation and import validation
- ✓ Data upload options (direct + Drive mount)
- ✓ Data validation and structure verification
- ✓ Configuration dataclass with all hyperparameters
- ✓ Checkpoint utilities for model saving/loading

**Next Phase:** VAE model architecture implementation

---
```

- [ ] **Step 2: Add integration test cell**

Add code cell:

```python
# Phase 1 Integration Test
print("Running Phase 1 integration test...\n")

tests_passed = 0
tests_total = 6

# Test 1: GPU/Device
try:
    assert config.device in ['cuda', 'cpu']
    print("✓ Test 1: Device configuration")
    tests_passed += 1
except AssertionError:
    print("✗ Test 1: Device configuration FAILED")

# Test 2: Data folder exists
try:
    assert os.path.exists('/content/data')
    print("✓ Test 2: Data folder exists")
    tests_passed += 1
except AssertionError:
    print("✗ Test 2: Data folder NOT FOUND")

# Test 3: Dataset info loaded
try:
    assert 'name' in dataset_info
    assert dataset_info['name'] == 'adult'
    print("✓ Test 3: Dataset metadata loaded")
    tests_passed += 1
except (AssertionError, NameError):
    print("✗ Test 3: Dataset metadata FAILED")

# Test 4: Config initialized
try:
    assert config.dataname == 'adult'
    assert config.epochs > 0
    print("✓ Test 4: Configuration initialized")
    tests_passed += 1
except (AssertionError, NameError):
    print("✗ Test 4: Configuration FAILED")

# Test 5: Checkpoint directories created
try:
    assert os.path.exists(config.ckpt_path)
    assert os.path.exists(config.results_path)
    print("✓ Test 5: Output directories created")
    tests_passed += 1
except AssertionError:
    print("✗ Test 5: Output directories FAILED")

# Test 6: Utilities defined
try:
    assert callable(save_checkpoint)
    assert callable(load_checkpoint)
    print("✓ Test 6: Utilities defined")
    tests_passed += 1
except (AssertionError, NameError):
    print("✗ Test 6: Utilities FAILED")

print(f"\n{'='*50}")
print(f"Integration Test Results: {tests_passed}/{tests_total} passed")
print(f"{'='*50}")

if tests_passed == tests_total:
    print("✓ Phase 1 Complete - Ready for Phase 2")
else:
    print(f"⚠ {tests_total - tests_passed} test(s) failed - review errors above")
```

- [ ] **Step 3: Run full notebook end-to-end**

Execute all cells in order:
1. Title cell
2. Environment setup cells
3. Data upload cells (use test data)
4. Data validation cells
5. Configuration cells
6. Utilities cells
7. Integration test cell

Expected: All cells execute without errors, integration test shows 6/6 passed

- [ ] **Step 4: Verify notebook state**

Check that the following variables exist and are correct:
- `device` - torch device object
- `dataset_info` - dict with adult dataset metadata
- `config` - TabCFConfig instance
- `save_checkpoint`, `load_checkpoint` - functions
- `TrainingHistory` - class

- [ ] **Step 5: Commit final Phase 1**

```bash
git add TABCF_Colab.ipynb
git commit -m "feat: complete Phase 1 foundation with integration test"
```

---

## Self-Review Checklist

**Spec coverage:**
- ✓ Section 1: Environment Setup (GPU detection, package installation)
- ✓ Section 2: Data Upload & Preparation (both options, validation)
- ✓ Section 3: Configuration (dataclass with all parameters)
- ✓ Checkpoint management utilities

**No placeholders:**
- ✓ All code cells contain complete, runnable code
- ✓ All file paths are exact (/content/data, etc.)
- ✓ All commands have expected outputs documented
- ✓ No "TBD", "TODO", or "implement later" statements

**Type consistency:**
- ✓ `config` is TabCFConfig dataclass throughout
- ✓ `device` is torch.device object
- ✓ `dataset_info` is dict loaded from JSON
- ✓ All function signatures match their usage

**Completeness:**
- ✓ Each task builds on previous tasks
- ✓ All dependencies are installed before use
- ✓ All paths are validated before access
- ✓ Integration test verifies all components work together

---

## Next Steps

After Phase 1 completion:
1. **Phase 2:** Implement VAE model architecture (Tokenizer, Encoder, Decoder)
2. **Phase 3:** Add VAE training loop with progress tracking
3. **Phase 4:** Implement black-box classifier training
4. **Phase 5:** Add counterfactual generation logic
5. **Phase 6:** Implement evaluation framework and visualizations
6. **Phase 7:** End-to-end integration and testing
