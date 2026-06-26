# TABCF Colab Migration - Phase 3: VAE Training Loop

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the VAE training loop with loss computation, optimization, progress tracking, and checkpointing.

**Architecture:** Add training cells including loss computation function, dataset handling, training loop with progress bars, and visualization of training curves.

**Tech Stack:** PyTorch training loop, tqdm for progress, matplotlib for visualization

---

## File Structure

**Files to modify:**
- `TABCF_Colab.ipynb` - Add VAE training cells (~6-8 new cells)

**Files to reference:**
- `tabcf/vae/main.py` - Source training logic and loss computation
- `utils_train.py` - Dataset and preprocessing utilities
- Design spec: `docs/superpowers/specs/2026-06-11-tabcf-colab-migration-design.md`

---

## Task 1: Add Loss Computation Function

**Files:**
- Modify: `TABCF_Colab.ipynb` (add cells)
- Reference: `tabcf/vae/main.py:38-94`

- [ ] **Step 1: Add training section header**

Add markdown cell:

```markdown
## 5. VAE Training

Train the transformer-based VAE with KL annealing and mixed reconstruction loss.
```

- [ ] **Step 2: Add loss computation header**

Add markdown cell:

```markdown
### Loss Computation

VAE loss = Reconstruction Loss + β × KL Divergence
- Numerical: L2 or L1 loss
- Categorical: L1 loss on Gumbel-Softmax outputs vs one-hot targets
- KL: -0.5 × mean(1 + logvar - mu² - exp(logvar))
```

- [ ] **Step 3: Add compute_loss function**

Add code cell (adapt from `tabcf/vae/main.py:38-94`):

```python
def compute_loss(X_num, X_cat, Recon_X_num, Recon_X_cat, mu_z, logvar_z, 
                 gumbel_softmax=False, num_reconstr_loss="L2", device="cpu"):
    """
    Compute VAE loss: reconstruction + KL divergence.
    
    Args:
        X_num: Original numerical features (batch, d_numerical)
        X_cat: Original categorical features (batch, n_categorical)
        Recon_X_num: Reconstructed numerical (batch, d_numerical)
        Recon_X_cat: List of reconstructed categorical probabilities
        mu_z: Latent mean (batch, d_latent)
        logvar_z: Latent log-variance (batch, d_latent)
        gumbel_softmax: Whether using Gumbel-Softmax (affects cat loss)
        num_reconstr_loss: "L2" or "L1" for numerical reconstruction
        device: torch device
    
    Returns:
        num_loss, cat_loss, loss_kld, acc
    """
    # Numerical reconstruction loss
    if X_num.shape[-1] == 0:
        num_loss = torch.tensor([0]).to(device)
    else:
        if num_reconstr_loss == "L2":
            num_loss = (X_num - Recon_X_num).pow(2).mean()
        else:
            num_loss = (X_num - Recon_X_num).abs().mean()
    
    # Categorical reconstruction loss
    cat_loss = 0
    acc = 0
    total_num = 0
    
    if gumbel_softmax:
        cat_loss_fn = nn.L1Loss()
        
        for idx, x_cat in enumerate(Recon_X_cat):
            if x_cat is not None:
                # Convert categorical labels to one-hot
                x_cat_ohe = F.one_hot(X_cat[:, idx], num_classes=x_cat.shape[-1]).float()
                cat_loss += cat_loss_fn(x_cat, x_cat_ohe)
                
                # Calculate accuracy
                x_hat_for_acc = x_cat.argmax(dim=-1)
                acc += (x_hat_for_acc == X_cat[:, idx]).float().sum()
                total_num += x_hat_for_acc.shape[0]
        
        cat_loss /= (idx + 1)
        acc /= total_num
    else:
        cat_loss_fn = nn.CrossEntropyLoss()
        
        for idx, x_cat in enumerate(Recon_X_cat):
            if x_cat is not None:
                cat_loss += cat_loss_fn(x_cat, X_cat[:, idx])
                
                x_hat = x_cat.argmax(dim=-1)
                acc += (x_hat == X_cat[:, idx]).float().sum()
                total_num += x_hat.shape[0]
        
        cat_loss /= (idx + 1)
        acc /= total_num
    
    # KL divergence loss
    temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()
    loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())
    
    return num_loss, cat_loss, loss_kld, acc

print("✓ Loss computation function defined")
```

- [ ] **Step 4: Test loss computation**

Add code cell:

```python
# Test loss computation
batch_size = 4
d_numerical = 5
categories = [3, 4, 2]
d_latent = 16

# Create dummy data
X_num = torch.randn(batch_size, d_numerical)
X_cat = torch.tensor([[0, 1, 0], [1, 2, 1], [2, 3, 0], [0, 0, 1]])
Recon_X_num = torch.randn(batch_size, d_numerical)
Recon_X_cat = [
    torch.randn(batch_size, 3).softmax(dim=-1),
    torch.randn(batch_size, 4).softmax(dim=-1),
    torch.randn(batch_size, 2).softmax(dim=-1)
]
mu_z = torch.randn(batch_size, d_latent)
logvar_z = torch.randn(batch_size, d_latent)

# Compute loss
num_loss, cat_loss, kl_loss, acc = compute_loss(
    X_num, X_cat, Recon_X_num, Recon_X_cat, mu_z, logvar_z,
    gumbel_softmax=True, num_reconstr_loss="L2", device="cpu"
)

print(f"Numerical loss: {num_loss.item():.4f}")
print(f"Categorical loss: {cat_loss:.4f}")
print(f"KL divergence: {kl_loss.item():.4f}")
print(f"Categorical accuracy: {acc:.4f}")
assert num_loss >= 0 and cat_loss >= 0 and kl_loss >= 0
assert 0 <= acc <= 1
print("✓ Loss computation test passed")
```

- [ ] **Step 5: Skip git commit**

(No commit - as requested)

---

## Task 2: Add Data Loading Utilities

**Files:**
- Modify: `TABCF_Colab.ipynb` (add cells)
- Reference: `utils_train.py`

- [ ] **Step 1: Add data loading header**

Add markdown cell:

```markdown
### Data Loading

Create PyTorch Dataset and DataLoader for training.
```

- [ ] **Step 2: Add TabularDataset class**

Add code cell:

```python
class TabularDataset(Dataset):
    """PyTorch Dataset for tabular data."""
    
    def __init__(self, X_num, X_cat, y=None):
        """
        Args:
            X_num: Numerical features (numpy array or None)
            X_cat: Categorical features (numpy array or None)
            y: Labels (numpy array or None)
        """
        self.X_num = torch.FloatTensor(X_num) if X_num is not None else None
        self.X_cat = torch.LongTensor(X_cat) if X_cat is not None else None
        self.y = torch.LongTensor(y) if y is not None else None
        
        # Determine length from whichever is not None
        if self.X_num is not None:
            self.length = len(self.X_num)
        elif self.X_cat is not None:
            self.length = len(self.X_cat)
        elif self.y is not None:
            self.length = len(self.y)
        else:
            raise ValueError("At least one of X_num, X_cat, or y must be provided")
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        item = {}
        if self.X_num is not None:
            item['X_num'] = self.X_num[idx]
        if self.X_cat is not None:
            item['X_cat'] = self.X_cat[idx]
        if self.y is not None:
            item['y'] = self.y[idx]
        return item

print("✓ TabularDataset class defined")
```

- [ ] **Step 3: Add data loading helper**

Add code cell:

```python
def load_adult_data():
    """Load preprocessed adult dataset from data folder."""
    adult_dir = config.data_path + '/adult'
    
    # Load numpy arrays
    X_num_train = np.load(os.path.join(adult_dir, 'X_num_train.npy'))
    X_cat_train = np.load(os.path.join(adult_dir, 'X_cat_train.npy'))
    y_train = np.load(os.path.join(adult_dir, 'y_train.npy'))
    
    X_num_test = np.load(os.path.join(adult_dir, 'X_num_test.npy'))
    X_cat_test = np.load(os.path.join(adult_dir, 'X_cat_test.npy'))
    y_test = np.load(os.path.join(adult_dir, 'y_test.npy'))
    
    print(f"Train set: {X_num_train.shape[0]} samples")
    print(f"Test set: {X_num_test.shape[0]} samples")
    print(f"Numerical features: {X_num_train.shape[1]}")
    print(f"Categorical features: {X_cat_train.shape[1]}")
    
    # Create datasets
    train_dataset = TabularDataset(X_num_train, X_cat_train, y_train)
    test_dataset = TabularDataset(X_num_test, X_cat_test, y_test)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False
    )
    
    # Get dataset info
    d_numerical = X_num_train.shape[1]
    categories = [len(np.unique(X_cat_train[:, i])) for i in range(X_cat_train.shape[1])]
    
    return train_loader, test_loader, d_numerical, categories

print("✓ Data loading function defined")
```

- [ ] **Step 4: Test data loading**

Add code cell:

```python
# Test data loading
train_loader, test_loader, d_numerical, categories = load_adult_data()

print(f"\nDataset info:")
print(f"  d_numerical: {d_numerical}")
print(f"  categories: {categories}")
print(f"  Train batches: {len(train_loader)}")
print(f"  Test batches: {len(test_loader)}")

# Test one batch
batch = next(iter(train_loader))
print(f"\nBatch shapes:")
print(f"  X_num: {batch['X_num'].shape}")
print(f"  X_cat: {batch['X_cat'].shape}")
print(f"  y: {batch['y'].shape}")

print("✓ Data loading test passed")
```

- [ ] **Step 5: Skip git commit**

(No commit - as requested)

---

## Task 3: Add KL Annealing Schedule

**Files:**
- Modify: `TABCF_Colab.ipynb` (add cells)

- [ ] **Step 1: Add annealing header**

Add markdown cell:

```markdown
### KL Annealing

Gradually increase KL weight (β) from min_beta to max_beta over epochs.
```

- [ ] **Step 2: Add annealing function**

Add code cell:

```python
def get_beta_schedule(epoch, total_epochs, min_beta, max_beta, annealing_type='linear'):
    """
    Compute β (KL weight) for current epoch.
    
    Args:
        epoch: Current epoch (0-indexed)
        total_epochs: Total training epochs
        min_beta: Starting β value
        max_beta: Final β value
        annealing_type: 'linear' or 'cosine'
    
    Returns:
        beta: KL weight for this epoch
    """
    if annealing_type == 'linear':
        # Linear annealing from min to max
        beta = min_beta + (max_beta - min_beta) * (epoch / total_epochs)
    elif annealing_type == 'cosine':
        # Cosine annealing
        progress = epoch / total_epochs
        beta = min_beta + (max_beta - min_beta) * (1 - math.cos(progress * math.pi)) / 2
    else:
        beta = max_beta
    
    return min(beta, max_beta)

print("✓ KL annealing function defined")
```

- [ ] **Step 3: Test annealing schedule**

Add code cell:

```python
# Test annealing schedule
epochs = [0, 25, 50, 75, 100]
betas_linear = [get_beta_schedule(e, 100, config.min_beta, config.max_beta, 'linear') for e in epochs]
betas_cosine = [get_beta_schedule(e, 100, config.min_beta, config.max_beta, 'cosine') for e in epochs]

print("Beta schedule (linear):")
for e, b in zip(epochs, betas_linear):
    print(f"  Epoch {e:3d}: β = {b:.6f}")

print("\nBeta schedule (cosine):")
for e, b in zip(epochs, betas_cosine):
    print(f"  Epoch {e:3d}: β = {b:.6f}")

# Visualize
import matplotlib.pyplot as plt
all_epochs = list(range(101))
all_betas_linear = [get_beta_schedule(e, 100, config.min_beta, config.max_beta, 'linear') for e in all_epochs]
all_betas_cosine = [get_beta_schedule(e, 100, config.min_beta, config.max_beta, 'cosine') for e in all_epochs]

plt.figure(figsize=(10, 4))
plt.plot(all_epochs, all_betas_linear, label='Linear', linewidth=2)
plt.plot(all_epochs, all_betas_cosine, label='Cosine', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('β (KL weight)')
plt.title('KL Annealing Schedule')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("✓ Annealing schedule test passed")
```

- [ ] **Step 4: Skip git commit**

(No commit - as requested)

---

## Task 4: Add Training Loop

**Files:**
- Modify: `TABCF_Colab.ipynb` (add cells)
- Reference: `tabcf/vae/main.py` training loop

- [ ] **Step 1: Add training loop header**

Add markdown cell:

```markdown
### Training Loop

Main training loop with progress tracking and checkpointing.
```

- [ ] **Step 2: Add train_vae function**

Add code cell:

```python
def train_vae(model, train_loader, test_loader, config):
    """
    Train the VAE model.
    
    Args:
        model: Model_VAE instance
        train_loader: Training DataLoader
        test_loader: Test DataLoader
        config: TabCFConfig instance
    
    Returns:
        history: TrainingHistory with loss curves
    """
    model = model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    history = TrainingHistory()
    best_loss = float('inf')
    
    print(f"\nTraining VAE for {config.epochs} epochs...")
    print(f"Device: {config.device}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Beta range: [{config.min_beta}, {config.max_beta}]")
    print("=" * 80)
    
    for epoch in range(config.epochs):
        # Get current beta (KL weight)
        beta = get_beta_schedule(epoch, config.epochs, config.min_beta, config.max_beta, 'linear')
        
        # Training phase
        model.train()
        train_num_loss = 0
        train_cat_loss = 0
        train_kl_loss = 0
        train_acc = 0
        train_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}", leave=False)
        for batch in pbar:
            X_num = batch['X_num'].to(config.device)
            X_cat = batch['X_cat'].to(config.device)
            
            # Forward pass
            Recon_X_num, Recon_X_cat, mu, logvar = model(X_num, X_cat)
            
            # Compute loss
            num_loss, cat_loss, kl_loss, acc = compute_loss(
                X_num, X_cat, Recon_X_num, Recon_X_cat, mu, logvar,
                gumbel_softmax=config.gumbel_softmax,
                num_reconstr_loss="L2",
                device=config.device
            )
            
            # Total loss with KL annealing
            total_loss = config.lambd * (num_loss + cat_loss) + beta * kl_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Accumulate metrics
            train_num_loss += num_loss.item()
            train_cat_loss += cat_loss
            train_kl_loss += kl_loss.item()
            train_acc += acc
            train_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'num': f'{num_loss.item():.4f}',
                'cat': f'{cat_loss:.4f}',
                'kl': f'{kl_loss.item():.4f}',
                'β': f'{beta:.6f}'
            })
        
        # Average training metrics
        train_num_loss /= train_batches
        train_cat_loss /= train_batches
        train_kl_loss /= train_batches
        train_acc /= train_batches
        train_total_loss = config.lambd * (train_num_loss + train_cat_loss) + beta * train_kl_loss
        
        # Validation phase
        model.eval()
        val_num_loss = 0
        val_cat_loss = 0
        val_kl_loss = 0
        val_acc = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch in test_loader:
                X_num = batch['X_num'].to(config.device)
                X_cat = batch['X_cat'].to(config.device)
                
                Recon_X_num, Recon_X_cat, mu, logvar = model(X_num, X_cat)
                
                num_loss, cat_loss, kl_loss, acc = compute_loss(
                    X_num, X_cat, Recon_X_num, Recon_X_cat, mu, logvar,
                    gumbel_softmax=config.gumbel_softmax,
                    num_reconstr_loss="L2",
                    device=config.device
                )
                
                val_num_loss += num_loss.item()
                val_cat_loss += cat_loss
                val_kl_loss += kl_loss.item()
                val_acc += acc
                val_batches += 1
        
        val_num_loss /= val_batches
        val_cat_loss /= val_batches
        val_kl_loss /= val_batches
        val_acc /= val_batches
        val_total_loss = config.lambd * (val_num_loss + val_cat_loss) + beta * val_kl_loss
        
        # Save history
        history.add(
            epoch=epoch,
            train_loss=train_total_loss,
            val_loss=val_total_loss,
            train_num_loss=train_num_loss,
            train_cat_loss=train_cat_loss,
            train_kl_loss=train_kl_loss,
            train_acc=train_acc,
            val_num_loss=val_num_loss,
            val_cat_loss=val_cat_loss,
            val_kl_loss=val_kl_loss,
            val_acc=val_acc,
            beta=beta
        )
        
        # Print epoch summary
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_total_loss:.4f} | Val Loss: {val_total_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} | β: {beta:.6f}")
        
        # Save best model
        if val_total_loss < best_loss:
            best_loss = val_total_loss
            checkpoint_path = os.path.join(config.ckpt_path, config.dataname, 'vae_model_best.pth')
            save_checkpoint(
                model, optimizer, epoch, val_total_loss, checkpoint_path,
                metadata={'beta': beta, 'val_acc': val_acc}
            )
        
        # Save periodic checkpoint
        if (epoch + 1) % config.save_interval == 0:
            checkpoint_path = os.path.join(config.ckpt_path, config.dataname, f'vae_model_epoch_{epoch+1}.pth')
            save_checkpoint(
                model, optimizer, epoch, val_total_loss, checkpoint_path,
                metadata={'beta': beta}
            )
    
    # Save final model
    final_path = os.path.join(config.ckpt_path, config.dataname, 'vae_model_final.pth')
    save_checkpoint(model, optimizer, config.epochs-1, val_total_loss, final_path)
    
    # Save training history
    history_path = os.path.join(config.ckpt_path, config.dataname, 'training_history.json')
    history.save(history_path)
    
    print("=" * 80)
    print(f"✓ Training complete!")
    print(f"  Best validation loss: {best_loss:.4f}")
    print(f"  Final validation loss: {val_total_loss:.4f}")
    print(f"  Checkpoints saved to: {config.ckpt_path}/{config.dataname}/")
    
    return history

print("✓ Training function defined")
```

- [ ] **Step 3: Skip git commit**

(No commit - as requested)

---

## Task 5: Run VAE Training

**Files:**
- Modify: `TABCF_Colab.ipynb` (add cells)

- [ ] **Step 1: Add training execution header**

Add markdown cell:

```markdown
### Run Training

Initialize model and start training (this will take 15-30 minutes on T4 GPU).
```

- [ ] **Step 2: Add model initialization and training cell**

Add code cell:

```python
# Initialize VAE model
print("Initializing VAE model...")
vae_model = Model_VAE(d_numerical, categories, config)

# Count parameters
total_params = sum(p.numel() for p in vae_model.parameters())
trainable_params = sum(p.numel() for p in vae_model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Train model
history = train_vae(vae_model, train_loader, test_loader, config)
```

- [ ] **Step 3: Skip git commit**

(No commit - as requested)

---

## Task 6: Add Training Visualization

**Files:**
- Modify: `TABCF_Colab.ipynb` (add cells)

- [ ] **Step 1: Add visualization header**

Add markdown cell:

```markdown
### Training Visualization

Plot loss curves and metrics over epochs.
```

- [ ] **Step 2: Add visualization cell**

Add code cell:

```python
# Plot training curves
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Total loss
axes[0, 0].plot(history.epochs, history.train_losses, label='Train', linewidth=2)
axes[0, 0].plot(history.epochs, history.val_losses, label='Val', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Total Loss')
axes[0, 0].set_title('Total Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Numerical loss
axes[0, 1].plot(history.epochs, history.metrics['train_num_loss'], label='Train', linewidth=2)
axes[0, 1].plot(history.epochs, history.metrics['val_num_loss'], label='Val', linewidth=2)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Numerical Loss')
axes[0, 1].set_title('Numerical Reconstruction Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Categorical loss
axes[0, 2].plot(history.epochs, history.metrics['train_cat_loss'], label='Train', linewidth=2)
axes[0, 2].plot(history.epochs, history.metrics['val_cat_loss'], label='Val', linewidth=2)
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('Categorical Loss')
axes[0, 2].set_title('Categorical Reconstruction Loss')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# KL divergence
axes[1, 0].plot(history.epochs, history.metrics['train_kl_loss'], label='Train', linewidth=2)
axes[1, 0].plot(history.epochs, history.metrics['val_kl_loss'], label='Val', linewidth=2)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('KL Divergence')
axes[1, 0].set_title('KL Divergence')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Categorical accuracy
axes[1, 1].plot(history.epochs, history.metrics['train_acc'], label='Train', linewidth=2)
axes[1, 1].plot(history.epochs, history.metrics['val_acc'], label='Val', linewidth=2)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].set_title('Categorical Reconstruction Accuracy')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Beta schedule
axes[1, 2].plot(history.epochs, history.metrics['beta'], linewidth=2, color='purple')
axes[1, 2].set_xlabel('Epoch')
axes[1, 2].set_ylabel('β (KL weight)')
axes[1, 2].set_title('KL Annealing Schedule')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("✓ Training visualization complete")
```

- [ ] **Step 3: Add Phase 3 completion cell**

Add markdown cell:

```markdown
---

## Phase 3 Complete ✓

**What we have:**
- ✓ Loss computation (reconstruction + KL divergence)
- ✓ Data loading with TabularDataset and DataLoader
- ✓ KL annealing schedule (linear/cosine)
- ✓ Complete training loop with progress tracking
- ✓ Checkpointing (best + periodic + final)
- ✓ Training history and visualization

**Next Phase:** Black-box classifier training

---
```

- [ ] **Step 4: Skip git commit**

(No commit - as requested)

---

## Self-Review Checklist

**Spec coverage:**
- ✓ Loss computation function (reconstruction + KL)
- ✓ Data loading utilities
- ✓ KL annealing schedule
- ✓ Training loop with optimizer
- ✓ Checkpointing and history tracking
- ✓ Training visualization

**No placeholders:**
- ✓ All functions have complete implementations
- ✓ All parameters documented
- ✓ Training loop handles edge cases
- ✓ Visualization covers all metrics

**Type consistency:**
- ✓ Loss functions return correct types
- ✓ Config usage matches previous phases
- ✓ Checkpoint format consistent with Phase 1 utilities

**Completeness:**
- ✓ Full training pipeline implemented
- ✓ Progress tracking with tqdm
- ✓ Gradient clipping for stability
- ✓ Best model selection and saving

---

## Next Steps

After Phase 3 completion:
1. **Phase 4:** Black-box classifier training
2. **Phase 5:** Counterfactual generation
3. **Phase 6:** Evaluation and visualization
4. **Phase 7:** End-to-end integration
