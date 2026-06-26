# TABCF Colab Migration - Phase 4: Black-Box Classifier Training

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the black-box MLP classifier that will be used as the prediction model for counterfactual generation.

**Architecture:** Add cells for black-box classifier model, training loop, evaluation metrics, and model saving.

**Tech Stack:** PyTorch MLP, sklearn metrics, matplotlib for visualization

---

## File Structure

**Files to modify:**
- `TABCF_Colab.ipynb` - Add classifier training cells (~6-8 new cells)

**Files to reference:**
- `tabcf/vae/train_black_box.py` - Black-box classifier training
- `tabcf/vae/model.py` - BBMLPCLF model class
- Design spec: `docs/superpowers/specs/2026-06-11-tabcf-colab-migration-design.md`

---

## Task 1: Add Black-Box Classifier Model

**Files:**
- Modify: `TABCF_Colab.ipynb` (add cells)
- Reference: `tabcf/vae/model.py` (BBMLPCLF class)

- [ ] **Step 1: Add classifier section header**

Add markdown cell:

```markdown
## 6. Black-Box Classifier Training

Train the MLP classifier that TABCF will generate counterfactuals for.
```

- [ ] **Step 2: Add classifier model header**

Add markdown cell:

```markdown
### Black-Box MLP Classifier

Simple 2-layer MLP for binary classification (income prediction).
```

- [ ] **Step 3: Add BBMLPCLF model class**

Add code cell:

```python
class BBMLPCLF(nn.Module):
    """Black-box MLP classifier for binary classification."""
    
    def __init__(self, input_dim, hidden_dim=16, dropout=0.5):
        super(BBMLPCLF, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        """
        Args:
            x: Input features (batch, input_dim) - concatenated [numerical, categorical_one_hot]
        Returns:
            logits: (batch, 1) - raw logits for binary classification
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def predict_proba(self, x):
        """Get probability predictions."""
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        return probs

print("✓ BBMLPCLF class defined")
```

- [ ] **Step 4: Test classifier model**

Add code cell:

```python
# Test classifier
input_dim = d_numerical + sum(categories)  # numerical + one-hot categorical
clf = BBMLPCLF(input_dim=input_dim, hidden_dim=config.hidden_dims)

# Create dummy input
batch_size = 4
dummy_input = torch.randn(batch_size, input_dim)

# Forward pass
logits = clf(dummy_input)
probs = clf.predict_proba(dummy_input)

print(f"Input shape: {dummy_input.shape}")
print(f"Logits shape: {logits.shape}")
print(f"Probabilities shape: {probs.shape}")
assert logits.shape == (batch_size, 1)
assert probs.shape == (batch_size, 1)
print("✓ Classifier test passed")
```

- [ ] **Step 5: Skip git commit**

(No commit - as requested)

---

## Task 2: Add Data Preparation for Classifier

**Files:**
- Modify: `TABCF_Colab.ipynb` (add cells)

- [ ] **Step 1: Add data preparation header**

Add markdown cell:

```markdown
### Data Preparation for Classifier

Convert categorical features to one-hot encoding for MLP input.
```

- [ ] **Step 2: Add data preparation function**

Add code cell:

```python
def prepare_classifier_data(X_num, X_cat, categories):
    """
    Prepare data for black-box classifier.
    Concatenate numerical features with one-hot encoded categorical features.
    
    Args:
        X_num: Numerical features (batch, d_numerical)
        X_cat: Categorical features (batch, n_categorical) - integer labels
        categories: List of category counts per feature
    
    Returns:
        X_combined: (batch, d_numerical + sum(categories))
    """
    batch_size = X_num.shape[0]
    
    # One-hot encode categorical features
    X_cat_onehot_list = []
    for i, n_categories in enumerate(categories):
        cat_col = X_cat[:, i]
        # One-hot encode
        onehot = F.one_hot(cat_col.long(), num_classes=n_categories).float()
        X_cat_onehot_list.append(onehot)
    
    # Concatenate all one-hot features
    if X_cat_onehot_list:
        X_cat_onehot = torch.cat(X_cat_onehot_list, dim=1)
        # Concatenate numerical and one-hot categorical
        X_combined = torch.cat([X_num, X_cat_onehot], dim=1)
    else:
        X_combined = X_num
    
    return X_combined

print("✓ Data preparation function defined")
```

- [ ] **Step 3: Test data preparation**

Add code cell:

```python
# Test data preparation with real data
batch = next(iter(train_loader))
X_num_batch = batch['X_num']
X_cat_batch = batch['X_cat']

X_combined = prepare_classifier_data(X_num_batch, X_cat_batch, categories)

print(f"Numerical features: {X_num_batch.shape}")
print(f"Categorical features: {X_cat_batch.shape}")
print(f"Combined features: {X_combined.shape}")
print(f"Expected dim: {d_numerical} + {sum(categories)} = {d_numerical + sum(categories)}")
assert X_combined.shape[1] == d_numerical + sum(categories)
print("✓ Data preparation test passed")
```

- [ ] **Step 4: Skip git commit**

(No commit - as requested)

---

## Task 3: Add Classifier Training Loop

**Files:**
- Modify: `TABCF_Colab.ipynb` (add cells)

- [ ] **Step 1: Add training loop header**

Add markdown cell:

```markdown
### Classifier Training Loop

Train the MLP classifier on the adult dataset.
```

- [ ] **Step 2: Add train_classifier function**

Add code cell:

```python
def train_classifier(model, train_loader, test_loader, categories, config):
    """
    Train black-box classifier.
    
    Args:
        model: BBMLPCLF instance
        train_loader: Training DataLoader
        test_loader: Test DataLoader
        categories: List of category counts
        config: TabCFConfig instance
    
    Returns:
        history: Training history dict
    """
    model = model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.clf_learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    print(f"\nTraining Classifier for {config.clf_epochs} epochs...")
    print(f"Device: {config.device}")
    print(f"Learning rate: {config.clf_learning_rate}")
    print("=" * 80)
    
    for epoch in range(config.clf_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.clf_epochs}", leave=False):
            X_num = batch['X_num'].to(config.device)
            X_cat = batch['X_cat'].to(config.device)
            y = batch['y'].to(config.device).float().unsqueeze(1)
            
            # Prepare input
            X_combined = prepare_classifier_data(X_num, X_cat, categories)
            
            # Forward pass
            logits = model(X_combined)
            loss = criterion(logits, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Metrics
            train_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (preds == y).sum().item()
            train_total += y.size(0)
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                X_num = batch['X_num'].to(config.device)
                X_cat = batch['X_cat'].to(config.device)
                y = batch['y'].to(config.device).float().unsqueeze(1)
                
                X_combined = prepare_classifier_data(X_num, X_cat, categories)
                logits = model(X_combined)
                loss = criterion(logits, y)
                
                val_loss += loss.item()
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)
        
        val_loss /= len(test_loader)
        val_acc = val_correct / val_total
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(config.ckpt_path, config.dataname, 'black_box_mlp_best.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc
            }, checkpoint_path)
            if config.verbose:
                print(f"  ✓ Best model saved (Val Acc: {val_acc:.4f})")
    
    # Save final model
    final_path = os.path.join(config.ckpt_path, config.dataname, 'black_box_mlp_final.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': config.clf_epochs - 1,
        'val_acc': val_acc
    }, final_path)
    
    print("=" * 80)
    print(f"✓ Training complete!")
    print(f"  Best validation accuracy: {best_val_acc:.4f}")
    print(f"  Final validation accuracy: {val_acc:.4f}")
    
    return history

print("✓ Classifier training function defined")
```

- [ ] **Step 3: Skip git commit**

(No commit - as requested)

---

## Task 4: Run Classifier Training

**Files:**
- Modify: `TABCF_Colab.ipynb` (add cells)

- [ ] **Step 1: Add training execution header**

Add markdown cell:

```markdown
### Run Classifier Training

Train the black-box classifier (takes ~2-5 minutes).
```

- [ ] **Step 2: Add training execution cell**

Add code cell:

```python
# Initialize classifier
input_dim = d_numerical + sum(categories)
black_box_model = BBMLPCLF(input_dim=input_dim, hidden_dim=config.hidden_dims, dropout=0.5)

print(f"Classifier parameters: {sum(p.numel() for p in black_box_model.parameters()):,}")

# Train classifier
clf_history = train_classifier(black_box_model, train_loader, test_loader, categories, config)
```

- [ ] **Step 3: Skip git commit**

(No commit - as requested)

---

## Task 5: Add Classifier Evaluation and Visualization

**Files:**
- Modify: `TABCF_Colab.ipynb` (add cells)

- [ ] **Step 1: Add evaluation header**

Add markdown cell:

```markdown
### Classifier Evaluation

Evaluate classifier performance with metrics and visualizations.
```

- [ ] **Step 2: Add evaluation and visualization cell**

Add code cell:

```python
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# Evaluate on test set
black_box_model.eval()
all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for batch in test_loader:
        X_num = batch['X_num'].to(config.device)
        X_cat = batch['X_cat'].to(config.device)
        y = batch['y'].to(config.device)
        
        X_combined = prepare_classifier_data(X_num, X_cat, categories)
        probs = black_box_model.predict_proba(X_combined)
        preds = (probs > 0.5).float().cpu()
        
        all_preds.append(preds)
        all_labels.append(y.cpu())
        all_probs.append(probs.cpu())

all_preds = torch.cat(all_preds).numpy().flatten()
all_labels = torch.cat(all_labels).numpy()
all_probs = torch.cat(all_probs).numpy().flatten()

# Classification report
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=['<=50K', '>50K']))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# ROC curve
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

# Visualizations
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Training curves
axes[0].plot(clf_history['train_loss'], label='Train', linewidth=2)
axes[0].plot(clf_history['val_loss'], label='Val', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Confusion matrix
im = axes[1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
axes[1].set_title('Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('True')
tick_marks = np.arange(2)
axes[1].set_xticks(tick_marks)
axes[1].set_yticks(tick_marks)
axes[1].set_xticklabels(['<=50K', '>50K'])
axes[1].set_yticklabels(['<=50K', '>50K'])
# Add text annotations
thresh = cm.max() / 2.
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

# ROC curve
axes[2].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
axes[2].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
axes[2].set_xlabel('False Positive Rate')
axes[2].set_ylabel('True Positive Rate')
axes[2].set_title('ROC Curve')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("✓ Classifier evaluation complete")
```

- [ ] **Step 3: Add Phase 4 completion cell**

Add markdown cell:

```markdown
---

## Phase 4 Complete ✓

**What we have:**
- ✓ Black-box MLP classifier model
- ✓ Data preparation (one-hot encoding for categorical features)
- ✓ Classifier training loop
- ✓ Model evaluation with confusion matrix and ROC curve
- ✓ Trained classifier ready for counterfactual generation

**Next Phase:** Counterfactual generation with TABCF

---
```

- [ ] **Step 4: Skip git commit**

(No commit - as requested)

---

## Self-Review Checklist

**Spec coverage:**
- ✓ Black-box classifier model (2-layer MLP)
- ✓ Data preparation for classifier input
- ✓ Training loop with progress tracking
- ✓ Model saving (best + final)
- ✓ Evaluation metrics and visualization

**No placeholders:**
- ✓ All functions have complete implementations
- ✓ All parameters documented
- ✓ Training loop handles both phases
- ✓ Visualization covers key metrics

**Type consistency:**
- ✓ Model interfaces match VAE expectations
- ✓ Data shapes consistent throughout
- ✓ Config usage matches previous phases

**Completeness:**
- ✓ Full training pipeline
- ✓ Proper evaluation metrics
- ✓ Best model selection and saving
- ✓ Ready for CF generation

---

## Next Steps

After Phase 4 completion:
1. **Phase 5:** Counterfactual generation using TABCF
2. **Phase 6:** Evaluation framework and baseline comparisons
3. **Phase 7:** End-to-end integration and final testing
