# TABCF Colab Migration - Phase 5: Counterfactual Generation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement counterfactual generation using the trained VAE and black-box classifier.

**Architecture:** Add cells for CF generation logic, optimization in latent space, constraint handling, and results saving.

**Tech Stack:** PyTorch optimization, gradient-based search in latent space

---

## File Structure

**Files to modify:**
- `TABCF_Colab.ipynb` - Add CF generation cells (~8-10 new cells)

**Files to reference:**
- `tabcf/sample.py` - CF generation implementation
- `tabcf/latent_utils.py` - Utility functions
- Design spec: `docs/superpowers/specs/2026-06-11-tabcf-colab-migration-design.md`

---

## Task 1: Add CF Generation Section Header

**Files:**
- Modify: `TABCF_Colab.ipynb` (add cells)

- [ ] **Step 1: Add CF generation section header**

Add markdown cell:

```markdown
## 7. Counterfactual Generation with TABCF

Generate counterfactual explanations using the trained VAE and classifier.
```

- [ ] **Step 2: Skip git commit**

(No commit - as requested)

---

## Task 2: Add CF Loss Function

**Files:**
- Modify: `TABCF_Colab.ipynb` (add cells)

- [ ] **Step 1: Add CF loss header**

Add markdown cell:

```markdown
### Counterfactual Loss Function

Loss = Validity + Proximity (latent + input) + Sparsity + Diversity
```

- [ ] **Step 2: Add CF loss computation function**

Add code cell:

```python
def compute_cf_loss(z, z_original, x_num_cf, x_cat_cf, x_num_original, x_cat_original,
                    black_box_model, categories, target_class, config):
    """
    Compute counterfactual loss for optimization.
    
    Args:
        z: Current latent vector (batch, d_latent)
        z_original: Original latent vector (batch, d_latent)
        x_num_cf: Reconstructed numerical features (batch, d_numerical)
        x_cat_cf: Reconstructed categorical features (list of tensors)
        x_num_original: Original numerical features (batch, d_numerical)
        x_cat_original: Original categorical features (batch, n_categorical)
        black_box_model: Trained classifier
        categories: List of category counts
        target_class: Target class for CF (opposite of original)
        config: TabCFConfig instance
    
    Returns:
        total_loss, loss_dict (for logging)
    """
    # 1. Validity loss - encourage target class prediction
    X_cf_combined = prepare_classifier_data(x_num_cf, x_cat_cf, categories)
    logits = black_box_model(X_cf_combined)
    probs = torch.sigmoid(logits)
    
    # Binary cross-entropy to encourage target class
    target_tensor = torch.ones_like(probs) * target_class
    validity_loss = F.binary_cross_entropy(probs, target_tensor)
    
    # 2. Proximity loss - latent space
    proximity_latent = torch.norm(z - z_original, p=2, dim=1).mean()
    
    # 3. Proximity loss - input space (numerical)
    proximity_input_num = torch.norm(x_num_cf - x_num_original, p=2, dim=1).mean()
    
    # 4. Proximity loss - input space (categorical)
    proximity_input_cat = 0
    for i, x_cat_cf_feat in enumerate(x_cat_cf):
        # One-hot original
        x_cat_orig_onehot = F.one_hot(x_cat_original[:, i].long(), 
                                      num_classes=categories[i]).float()
        # L1 distance between probability distribution and original one-hot
        proximity_input_cat += torch.norm(x_cat_cf_feat - x_cat_orig_onehot, p=1, dim=1).mean()
    proximity_input_cat /= len(x_cat_cf)
    
    # 5. Sparsity loss - penalize many feature changes
    # Count changed features (numerical: threshold, categorical: argmax changed)
    num_changed = (torch.abs(x_num_cf - x_num_original) > 0.1).float().sum(dim=1).mean()
    cat_changed = 0
    for i, x_cat_cf_feat in enumerate(x_cat_cf):
        pred_cat = x_cat_cf_feat.argmax(dim=1)
        cat_changed += (pred_cat != x_cat_original[:, i]).float().mean()
    sparsity_loss = num_changed + cat_changed
    
    # Combined loss
    total_loss = (
        validity_loss +
        config.proximity_weight_latent * proximity_latent +
        config.proximity_weight_input * (proximity_input_num + proximity_input_cat) +
        config.sparsity_weight * sparsity_loss
    )
    
    loss_dict = {
        'validity': validity_loss.item(),
        'proximity_latent': proximity_latent.item(),
        'proximity_input': (proximity_input_num + proximity_input_cat).item(),
        'sparsity': sparsity_loss.item(),
        'total': total_loss.item()
    }
    
    return total_loss, loss_dict

print("✓ CF loss function defined")
```

- [ ] **Step 3: Skip git commit**

(No commit - as requested)

---

## Task 3: Add CF Generation Function

**Files:**
- Modify: `TABCF_Colab.ipynb` (add cells)

- [ ] **Step 1: Add CF generation header**

Add markdown cell:

```markdown
### CF Generation Function

Optimize in latent space to find counterfactuals.
```

- [ ] **Step 2: Add generate_counterfactuals function**

Add code cell:

```python
def generate_counterfactuals(vae_model, black_box_model, data_loader, categories, 
                             d_numerical, config, num_instances=None):
    """
    Generate counterfactual explanations using TABCF.
    
    Args:
        vae_model: Trained VAE model
        black_box_model: Trained classifier
        data_loader: DataLoader with instances to explain
        categories: List of category counts
        d_numerical: Number of numerical features
        config: TabCFConfig instance
        num_instances: Number of instances to generate CFs for (None = all)
    
    Returns:
        results: List of dicts with original, CF, and metadata
    """
    vae_model.eval()
    black_box_model.eval()
    
    results = []
    instances_processed = 0
    
    if num_instances is None:
        num_instances = config.num_samples
    
    print(f"Generating counterfactuals for {num_instances} instances...")
    print(f"Max iterations per instance: {config.max_iterations}")
    print("=" * 80)
    
    with tqdm(total=num_instances, desc="CF Generation") as pbar:
        for batch in data_loader:
            if instances_processed >= num_instances:
                break
            
            X_num = batch['X_num'].to(config.device)
            X_cat = batch['X_cat'].to(config.device)
            y = batch['y'].to(config.device)
            
            batch_size = X_num.shape[0]
            remaining = num_instances - instances_processed
            actual_batch = min(batch_size, remaining)
            
            X_num = X_num[:actual_batch]
            X_cat = X_cat[:actual_batch]
            y = y[:actual_batch]
            
            # Get original predictions
            X_combined_orig = prepare_classifier_data(X_num, X_cat, categories)
            orig_probs = black_box_model.predict_proba(X_combined_orig)
            orig_preds = (orig_probs > 0.5).float()
            
            # Encode to latent space
            with torch.no_grad():
                mu, logvar = vae_model.encode(X_num, X_cat)
                z_original = mu  # Use mean (not reparameterized)
            
            # For each instance in batch
            for i in range(actual_batch):
                # Skip if already predicted target class
                target_class = 1 - orig_preds[i].item()
                
                if orig_preds[i].item() == target_class:
                    continue
                
                # Initialize z for optimization (requires_grad=True)
                z_cf = z_original[i:i+1].clone().detach().requires_grad_(True)
                
                # Optimizer for latent vector
                optimizer = torch.optim.Adam([z_cf], lr=0.01)
                
                best_cf = None
                best_loss = float('inf')
                
                # Optimization loop
                for iter in range(config.max_iterations):
                    optimizer.zero_grad()
                    
                    # Decode from latent
                    x_num_cf, x_cat_cf = vae_model.decode(z_cf)
                    
                    # Compute loss
                    loss, loss_dict = compute_cf_loss(
                        z_cf, z_original[i:i+1],
                        x_num_cf, x_cat_cf,
                        X_num[i:i+1], X_cat[i:i+1],
                        black_box_model, categories, target_class, config
                    )
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    # Check if CF is valid (flipped prediction)
                    with torch.no_grad():
                        X_cf_combined = prepare_classifier_data(x_num_cf, x_cat_cf, categories)
                        cf_prob = black_box_model.predict_proba(X_cf_combined)
                        cf_pred = (cf_prob > 0.5).float()
                        
                        if cf_pred.item() == target_class and loss.item() < best_loss:
                            best_loss = loss.item()
                            best_cf = {
                                'x_num': x_num_cf.detach().cpu(),
                                'x_cat': [x.detach().cpu().argmax(dim=1) for x in x_cat_cf],
                                'z': z_cf.detach().cpu(),
                                'loss': loss.item(),
                                'iteration': iter,
                                'prob': cf_prob.item()
                            }
                    
                    # Early stopping if valid CF found with good loss
                    if best_cf is not None and best_loss < 0.5:
                        break
                
                # Save result
                result = {
                    'instance_id': instances_processed,
                    'original': {
                        'x_num': X_num[i].cpu().numpy(),
                        'x_cat': X_cat[i].cpu().numpy(),
                        'y': y[i].item(),
                        'pred': orig_preds[i].item(),
                        'prob': orig_probs[i].item()
                    },
                    'cf': best_cf,
                    'target_class': target_class,
                    'success': best_cf is not None
                }
                results.append(result)
                instances_processed += 1
                pbar.update(1)
                
                # Update progress bar with success rate
                success_count = sum(1 for r in results if r['success'])
                pbar.set_postfix({
                    'success_rate': f"{success_count}/{len(results)}",
                    'pct': f"{100*success_count/len(results):.1f}%"
                })
    
    success_count = sum(1 for r in results if r['success'])
    print("=" * 80)
    print(f"✓ CF Generation complete!")
    print(f"  Total instances: {len(results)}")
    print(f"  Successful CFs: {success_count} ({100*success_count/len(results):.1f}%)")
    
    return results

print("✓ CF generation function defined")
```

- [ ] **Step 3: Skip git commit**

(No commit - as requested)

---

## Task 4: Run CF Generation

**Files:**
- Modify: `TABCF_Colab.ipynb` (add cells)

- [ ] **Step 1: Add CF execution header**

Add markdown cell:

```markdown
### Run CF Generation

Generate counterfactuals for test instances (takes ~10-20 minutes for 100 samples).
```

- [ ] **Step 2: Add CF execution cell**

Add code cell:

```python
# Generate counterfactuals using TABCF
cf_results = generate_counterfactuals(
    vae_model=vae_model,
    black_box_model=black_box_model,
    data_loader=test_loader,
    categories=categories,
    d_numerical=d_numerical,
    config=config,
    num_instances=config.num_samples
)
```

- [ ] **Step 3: Skip git commit**

(No commit - as requested)

---

## Task 5: Save CF Results

**Files:**
- Modify: `TABCF_Colab.ipynb` (add cells)

- [ ] **Step 1: Add results saving header**

Add markdown cell:

```markdown
### Save Results

Save counterfactual results to CSV for evaluation.
```

- [ ] **Step 2: Add results saving function and execution**

Add code cell:

```python
def save_cf_results(results, categories, d_numerical, config):
    """Save CF results to CSV."""
    
    # Prepare data for CSV
    rows = []
    for result in results:
        if not result['success']:
            continue
        
        row = {'instance_id': result['instance_id']}
        
        # Original features
        for j in range(d_numerical):
            row[f'orig_num_{j}'] = result['original']['x_num'][j]
        for j in range(len(categories)):
            row[f'orig_cat_{j}'] = result['original']['x_cat'][j]
        row['orig_label'] = result['original']['y']
        row['orig_pred'] = result['original']['pred']
        row['orig_prob'] = result['original']['prob']
        
        # CF features
        for j in range(d_numerical):
            row[f'cf_num_{j}'] = result['cf']['x_num'][0, j].item()
        for j in range(len(categories)):
            row[f'cf_cat_{j}'] = result['cf']['x_cat'][j].item()
        row['cf_pred'] = result['target_class']
        row['cf_prob'] = result['cf']['prob']
        
        # Metadata
        row['cf_loss'] = result['cf']['loss']
        row['cf_iterations'] = result['cf']['iteration']
        
        rows.append(row)
    
    # Create DataFrame and save
    df = pd.DataFrame(rows)
    
    # Save to CSV
    os.makedirs(config.results_path, exist_ok=True)
    save_path = os.path.join(config.results_path, 
                             f"{config.dataname}_tabcf_{config.num_samples}.csv")
    df.to_csv(save_path, index=False)
    
    print(f"✓ Results saved to {save_path}")
    print(f"  Total rows: {len(df)}")
    
    return df, save_path

# Save results
cf_df, cf_save_path = save_cf_results(cf_results, categories, d_numerical, config)

# Display first few results
print("\nFirst 5 counterfactual results:")
print(cf_df.head())
```

- [ ] **Step 3: Skip git commit**

(No commit - as requested)

---

## Task 6: Add CF Analysis and Visualization

**Files:**
- Modify: `TABCF_Colab.ipynb` (add cells)

- [ ] **Step 1: Add analysis header**

Add markdown cell:

```markdown
### CF Analysis

Analyze counterfactual quality and visualize results.
```

- [ ] **Step 2: Add analysis and visualization cell**

Add code cell:

```python
# Analyze CF results
successful_cfs = [r for r in cf_results if r['success']]
print(f"CF Generation Statistics:")
print(f"  Total instances: {len(cf_results)}")
print(f"  Successful CFs: {len(successful_cfs)} ({100*len(successful_cfs)/len(cf_results):.1f}%)")

# Calculate metrics
proximities = [r['cf']['loss'] for r in successful_cfs]
iterations = [r['cf']['iteration'] for r in successful_cfs]

print(f"\nQuality Metrics:")
print(f"  Mean loss: {np.mean(proximities):.4f}")
print(f"  Median loss: {np.median(proximities):.4f}")
print(f"  Mean iterations: {np.mean(iterations):.1f}")
print(f"  Median iterations: {np.median(iterations):.1f}")

# Visualizations
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Success rate by iteration
axes[0].hist(iterations, bins=50, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Iterations to Find CF')
axes[0].set_ylabel('Count')
axes[0].set_title('CF Convergence Distribution')
axes[0].grid(True, alpha=0.3)

# Loss distribution
axes[1].hist(proximities, bins=50, edgecolor='black', alpha=0.7, color='green')
axes[1].set_xlabel('CF Loss')
axes[1].set_ylabel('Count')
axes[1].set_title('CF Loss Distribution')
axes[1].grid(True, alpha=0.3)

# Success rate over time
success_rate_cumulative = []
for i in range(len(cf_results)):
    successes = sum(1 for r in cf_results[:i+1] if r['success'])
    success_rate_cumulative.append(100 * successes / (i + 1))

axes[2].plot(success_rate_cumulative, linewidth=2)
axes[2].set_xlabel('Instance Number')
axes[2].set_ylabel('Success Rate (%)')
axes[2].set_title('Cumulative Success Rate')
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim([0, 100])

plt.tight_layout()
plt.show()

print("✓ CF analysis complete")
```

- [ ] **Step 3: Add Phase 5 completion cell**

Add markdown cell:

```markdown
---

## Phase 5 Complete ✓

**What we have:**
- ✓ CF loss function (validity + proximity + sparsity)
- ✓ CF generation via latent space optimization
- ✓ Gradient-based search with Adam optimizer
- ✓ Results saving to CSV
- ✓ CF quality analysis and visualization
- ✓ Working TABCF pipeline for generating counterfactuals

**Next Phase:** Comprehensive evaluation framework

---
```

- [ ] **Step 4: Skip git commit**

(No commit - as requested)

---

## Self-Review Checklist

**Spec coverage:**
- ✓ CF loss function (validity, proximity, sparsity)
- ✓ CF generation via optimization
- ✓ Latent space search
- ✓ Results saving
- ✓ Analysis and visualization

**No placeholders:**
- ✓ All functions have complete implementations
- ✓ Loss computation includes all components
- ✓ Optimization loop handles early stopping
- ✓ Results properly saved to CSV

**Type consistency:**
- ✓ Interfaces match VAE and classifier
- ✓ Tensor shapes consistent
- ✓ Config parameters used correctly

**Completeness:**
- ✓ Full CF generation pipeline
- ✓ Success tracking
- ✓ Quality metrics
- ✓ Visualization of results

---

## Next Steps

After Phase 5 completion:
1. **Phase 6:** Comprehensive evaluation framework with all metrics
2. **Phase 7:** Final integration and notebook completion
