
# TABCF Google Colab Migration Design

**Date:** 2026-06-11  
**Goal:** Migrate TABCF counterfactual explanation research codebase to Google Colab for full experimental workflow (training from scratch, CF generation, evaluation)

## Overview

Convert the TABCF project into a single comprehensive Jupyter notebook that runs on Google Colab, enabling researchers to train models, generate counterfactual explanations, and evaluate results entirely in the browser with GPU acceleration.

## User Requirements

- **Primary Goal:** Run full research experiments from scratch (train VAE → train classifier → generate CFs → evaluate)
- **Dataset:** Local `data/` folder (333MB) with preprocessed adult dataset already available
- **Training:** Train models from scratch in Colab (no pre-trained model reuse)
- **Use Case:** Academic research workflow with experimentation and reproducibility

## Architecture Decision: Single Monolithic Notebook

**Selected Approach:** One comprehensive notebook with all functionality organized in sections

**Rationale:**
- Easiest to share and reproduce for research purposes
- Cell-by-cell execution enables inspection of intermediate results
- Perfect for experimentation and hyperparameter tuning
- Can run partial workflows (just training, just sampling, etc.)
- Most Colab-native experience

**Alternatives Considered:**
- Multi-notebook pipeline: Better for production but adds complexity for research
- Code upload + thin notebook: Preserves structure but less notebook-friendly

## Notebook Structure

The notebook consists of 7 main sections executed sequentially:

### 1. Environment Setup
- Install dependencies from `requirements-kaggle.txt` (optimized for Colab)
- Auto-detect GPU availability (`torch.cuda.is_available()`)
- Configure `torch.device` (CUDA if available, CPU fallback with warning)
- Verify PyTorch >= 2.0, TensorFlow installation
- Import validation (fail fast on missing packages)

**Dependency handling:**
- Leverage pre-installed PyTorch/TensorFlow in Colab
- Install: transformers, datasets, peft, ml_collections, sdmetrics, prdc, xgboost, category_encoders, imbalanced-learn
- DiCE framework: Extract optimization logic directly into notebook cells (avoid full package installation)
- CARLA framework: Skip unless baseline comparisons needed (reduces dependencies)

**Optional features:**
- Google Drive mount for persistent storage of checkpoints and results
- Drive mounting instructions for faster repeated runs

### 2. Data Upload & Preparation

**Two options presented to user:**

**Option A - Direct Upload (default for first run):**
```python
from google.colab import files
# Upload data.zip → extract to /content/data/
# Estimated time: 2-3 minutes for 333MB
```

**Option B - Google Drive Mount (for repeated runs):**
```python
from google.colab import drive
drive.mount('/content/drive')
# Copy or symlink from Drive to /content/data/
```

**Data validation:**
- Check for required folder structure: `data/Info/`, `data/adult/`
- Verify existence of key files: `info.json`, `.npy` arrays, `black_box_mlp_hidden_16.pkl`
- Display dataset statistics (train/test split, feature counts, class balance)

**File organization in Colab:**
```
/content/
├── data/                    # Uploaded data folder
│   ├── Info/               # Dataset configs (adult.json, etc.)
│   └── adult/              # Preprocessed numpy arrays, models
├── ckpt/                   # Model checkpoints (created during training)
└── counterfactual_results/ # Generated CFs (created during sampling)
```

### 3. Configuration

**Configuration dataclass** defining all hyperparameters and paths:

```python
@dataclass
class TabCFConfig:
    # Dataset
    dataname: str = 'adult'  # Easy to switch datasets
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # VAE Training
    max_beta: float = 1e-2    # KL weight annealing max
    min_beta: float = 1e-5    # KL weight annealing min
    lambd: float = 0.7        # Balance reconstruction vs KL
    gumbel_softmax: bool = True  # Differentiable categorical
    tau: float = 1.0          # Gumbel-Softmax temperature
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-3
    
    # Black-box Classifier
    hidden_dims: int = 16     # MLP hidden layer size
    
    # CF Generation
    num_samples: int = 100    # Test instances to explain
    proximity_weight_input: float = 1.0
    proximity_weight_latent: float = 1.0
    diversity_weight: float = 1.0
    sparsity_weight: float = 0.5
    max_iterations: int = 1000
    
    # Paths
    data_path: str = '/content/data'
    ckpt_path: str = '/content/ckpt'
    results_path: str = '/content/counterfactual_results'
```

Users can modify configuration in-cell before training.

### 4. VAE Training

**Implementation of transformer-based VAE from `tabcf/vae/`:**

**Embedded code cells:**
- VAE model architecture (transformer encoder, Gumbel-Softmax detokenizer)
- Training loop with progress tracking
- KL weight annealing schedule (min_beta → max_beta)
- Mixed reconstruction loss (continuous: L2, categorical: cross-entropy)

**Training features:**
- Progress bars showing: epoch, reconstruction loss, KL divergence, total loss
- Inline loss curve plots (updated every 10 epochs)
- Automatic checkpoint saving: `ckpt/adult/vae_model_best.pth`, `vae_model_final.pth`
- Early stopping option based on validation loss
- Training history saved to JSON

**Estimated runtime:** 15-30 minutes on Colab T4 GPU for adult dataset

**Why this design:**
The transformer-based VAE learns a continuous latent space suitable for gradient-based CF optimization. The Gumbel-Softmax detokenizer enables differentiable categorical reconstruction, critical for mixed tabular data.

### 5. Black-Box Model Training

**Train MLP classifier** (the model we'll explain with counterfactuals):

**Implementation:**
- 2-layer MLP with configurable hidden dimensions (default: 16)
- Binary classification for income prediction
- Trained on processed data (X_num_train.npy, X_cat_train.npy)
- Model saved to `ckpt/adult/black_box_mlp.pkl`

**Training features:**
- Training/validation accuracy tracking
- Confusion matrix visualization
- ROC curve and AUC score
- Model performance summary

**Estimated runtime:** 2-5 minutes

**Why this design:**
The black-box classifier represents the production model we want to explain. TABCF generates CFs that flip this model's predictions.

### 6. Counterfactual Generation

**Primary method:** TABCF using trained VAE

**Generation workflow:**
1. Load trained VAE and black-box classifier from checkpoints
2. Select test instances (default: 100 random from test set)
3. For each instance:
   - Encode to latent space
   - Optimize latent vector via gradient descent to flip prediction
   - Decode back to input space
   - Validate CF (prediction changed, constraints satisfied)
4. Save results to CSV

**Optimization in latent space:**
- Loss function: `λ₁·proximity_latent + λ₂·proximity_input + λ₃·diversity + λ₄·sparsity + validity_loss`
- Gradient-based search using Adam optimizer
- Early stopping when valid CF found or max_iterations reached

**Progress tracking:**
- Progress bar: CF generation per instance
- Success rate counter (% valid CFs found)
- Average optimization time per instance
- Live metric updates

**Output format:**
CSV saved to `/content/counterfactual_results/adult_tabcf_100.csv`:

| instance_id | age | workclass | ... | income | cf_age | cf_workclass | ... | cf_income | distance | sparsity |
|-------------|-----|-----------|-----|--------|--------|--------------|-----|-----------|----------|----------|

**Baseline methods (optional collapsible sections):**
- **DiCE:** Diverse Counterfactual Explanations (gradient-based optimization)
- **Wachter:** Direct gradient descent in input space
- **REVISE:** Recourse via Variational Inference
- **CCHVAE:** Conditional VAE approach

Each baseline method in its own collapsible cell section for optional comparison.

**Why this design:**
Optimization in the learned latent space (vs. raw input space) ensures generated CFs lie on the data manifold and respect feature correlations learned by the VAE.

### 7. Evaluation & Visualization

**Evaluation framework** calculating all paper metrics:

**Primary Metrics:**
1. **Validity** - % of CFs that successfully flip the prediction
2. **Proximity** - Average distance from original (L1, L2, weighted)
   - Input space distance: `||x - x'||`
   - Latent space distance: `||z - z'||`
3. **Sparsity** - Average number of features changed
4. **Data Manifold Closeness** - Outlier detection score (lower = more realistic)

**Constraint Metrics:**
5. **Immutability Violation** - % CFs changing protected features (race, sex, marital status)
6. **Monotonicity** - % CFs respecting monotonic relationships (e.g., education ↑ → income ↑)

**Diversity Metrics (when multiple CFs per instance):**
7. **Diversity** - Average pairwise distance between CFs
8. **Coverage** - Variety of feature change patterns

**Additional Metrics:**
9. **Success Rate** - % of instances where valid CF found
10. **Optimization Time** - Average seconds per instance

**Evaluation workflow:**
```python
# Automatic evaluation from saved CSV
results_df = evaluate_counterfactuals(
    dataname='adult',
    method='tabcf',
    num_samples=100
)
# Returns DataFrame with all metrics per instance + aggregate statistics
```

**Visualizations:**

**1. Training Diagnostics:**
- VAE loss curves (reconstruction, KL, total loss over epochs)
- Beta annealing schedule visualization
- Black-box classifier performance (accuracy, ROC curve)

**2. CF Quality Analysis:**
- Proximity vs Sparsity scatter plot (trade-off visualization)
- Validity rate across methods (bar chart: TABCF vs baselines)
- Feature change frequency histogram (which features changed most)
- Distance distribution (violin plot showing proximity spread)

**3. Feature-Level Analysis:**
- Per-feature change statistics (mean, std of changes)
- Immutable feature violation tracking
- Feature correlation preservation (before/after CF generation)

**4. Comparison Tables:**
- Side-by-side metric comparison (TABCF vs baselines)
- Statistical significance tests (paired t-test for proximity, validity)
- Ranking table across all metrics

**5. Example Showcases:**
- Display 5-10 example original → CF pairs in readable tables
- Highlight changed features in color
- Show prediction probabilities (original vs CF)
- Include distance and sparsity for each example

**Interactive Elements:**
- Adjust metric weights and recalculate scores
- Filter results by validity/distance thresholds
- Export summary statistics to CSV
- Save visualizations as PNG files

**Why this design:**
Comprehensive evaluation enables direct comparison with paper results and baselines. Visualizations make CF quality interpretable for non-technical stakeholders.

## Code Embedding Strategy

**Approach:** Embed Python modules directly as notebook cells (no external .py files)

**Cell organization:**
1. Imports and helpers (utils.py functions)
2. Configuration dataclass
3. Data loading utilities
4. VAE model architecture (tabcf/vae/model.py)
5. VAE training logic (tabcf/vae/main.py)
6. Black-box model class
7. Black-box training logic
8. TABCF sampling (tabcf/sample.py)
9. Baseline methods (baselines/*/sample.py) - optional
10. Evaluation framework (evaluation_framework/evaluate.py)
11. Visualization utilities

**Benefits:**
- No file system dependencies (pure notebook)
- Can modify code inline for experimentation
- All logic visible in one place
- Each major class/function in its own cell for modularity

**How to apply:**
Each original .py module becomes a set of notebook cells with clear headers (e.g., "# VAE Model Architecture"). Use collapsible sections for optional components (baselines).

## Checkpoint Management

**Checkpoint structure:**
```
/content/ckpt/{dataname}/
├── vae_model_best.pth       # Best VAE (lowest validation loss)
├── vae_model_final.pth      # Final epoch
├── black_box_mlp.pkl        # Trained classifier
└── training_history.json    # Loss curves, hyperparameters, timestamps
```

**Resume capability:**
- Check for existing checkpoints at start of training sections
- Offer user choice: "Load existing checkpoint or retrain from scratch?"
- Display checkpoint metadata (date, epoch, validation loss)

**Drive persistence (optional):**
If Drive mounted, offer to save checkpoints to Drive for persistence across sessions:
```
/content/drive/MyDrive/TABCF_checkpoints/adult/
```

**Why this design:**
Checkpointing enables iterative experimentation without retraining. Drive persistence ensures work survives Colab session timeouts.

## Error Handling & Validation

**Defensive checks throughout:**

1. **Environment validation:**
   - GPU availability warning if CPU-only
   - PyTorch/CUDA version compatibility check
   - Memory estimation (warn if dataset + model exceeds available RAM)

2. **Data validation:**
   - File existence checks before loading
   - Array shape validation (train/test splits match expected dimensions)
   - Missing value detection
   - Feature type consistency (numeric vs categorical)

3. **Training stability:**
   - Gradient clipping to prevent exploding gradients
   - NaN loss detection (stop and alert user)
   - Checkpoint corruption detection (verify loadable)

4. **CF generation:**
   - Timeout per instance (skip if optimization stalls)
   - Invalid CF detection (constraints violated)
   - Success rate monitoring (alert if < 50%)

5. **Evaluation:**
   - Empty results handling (skip evaluation if no valid CFs)
   - Metric calculation failures (e.g., outlier model missing)

**User-facing error messages:**
- Clear, actionable error descriptions
- Suggestions for fixing common issues
- Links to relevant troubleshooting sections

**Why this design:**
Research code often fails silently. Explicit validation prevents wasted GPU hours and unclear results.

## Performance Optimization

**GPU utilization:**
- Batch processing where possible (VAE training, CF generation)
- Mixed precision training (torch.cuda.amp) if available
- DataLoader with multi-worker prefetching

**Memory management:**
- Checkpoint deletion after loading (free GPU memory)
- Gradient checkpointing for large models
- Clear CUDA cache between training stages

**Expected runtimes (Colab T4 GPU):**
- Environment setup: 2-3 minutes
- Data upload: 2-3 minutes (direct) or 10 seconds (Drive)
- VAE training: 15-30 minutes
- Black-box training: 2-5 minutes
- CF generation (100 samples): 10-20 minutes
- Evaluation: 2-5 minutes
- **Total:** ~35-70 minutes for full pipeline

**Why this design:**
Free Colab sessions have time limits. Optimizations ensure full workflow completes within one session.

## Extensibility

**Easy dataset switching:**
Change one line: `config.dataname = 'bank'` (requires uploading corresponding data folder)

**Hyperparameter tuning:**
Configuration dataclass makes it trivial to modify parameters and rerun training.

**New baseline methods:**
Add new collapsible section with method-specific CF generation logic.

**Custom metrics:**
Evaluation framework designed to accept additional metric functions.

**Export for production:**
Users can extract trained models and CF generation code for deployment outside Colab.

**Why this design:**
Researchers often need to experiment with variations. Modular design supports rapid iteration.

## Limitations & Trade-offs

**Notebook approach limitations:**
- Large file size (~500+ lines) - may be slow to load
- Less git-friendly than modular Python files
- No integrated testing framework (must validate manually)
- Colab session timeouts require re-uploading data

**Data size constraints:**
- 333MB upload is manageable, but larger datasets (>1GB) should use Drive mount
- Colab RAM limits (~12GB on free tier) may constrain batch sizes for large datasets

**Training time:**
- VAE training from scratch takes 15-30 minutes - no way around this
- Free Colab has usage limits (GPU hours per day)

**Baseline method coverage:**
- Full CARLA framework skipped to reduce dependencies
- Only core baselines included (DiCE, Wachter, REVISE, CCHVAE)

**Reproducibility:**
- Random seeds set, but Colab environment updates may cause slight variations
- GPU availability not guaranteed on free tier

**Why we accept these:**
The single-notebook approach prioritizes ease of use and shareability over perfect modularity. For research purposes, this trade-off is appropriate.

## Success Criteria

The migration is successful if:

1. ✅ Notebook runs end-to-end without manual file system setup
2. ✅ VAE and black-box models train successfully on adult dataset
3. ✅ TABCF generates valid counterfactuals (>80% validity rate)
4. ✅ Evaluation metrics match original codebase results (within 5% tolerance)
5. ✅ Total runtime fits within one free Colab session (<3 hours)
6. ✅ Visualization plots display correctly inline
7. ✅ User can easily modify hyperparameters and rerun experiments
8. ✅ Checkpoint saving/loading works correctly
9. ✅ Error messages are clear and actionable
10. ✅ Notebook is shareable and reproducible by others

## Implementation Plan Next Steps

After design approval:
1. Write detailed implementation plan (invoke `writing-plans` skill)
2. Plan will break implementation into discrete tasks:
   - Cell-by-cell notebook construction
   - Code adaptation from .py modules
   - Testing each section independently
   - End-to-end validation
3. Implementation will produce final `.ipynb` file ready for upload to Colab
