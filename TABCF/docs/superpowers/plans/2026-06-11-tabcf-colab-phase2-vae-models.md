# TABCF Colab Migration - Phase 2: VAE Model Architecture

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the transformer-based VAE model architecture including Tokenizer, Encoder, Decoder, and supporting components.

**Architecture:** Add VAE model classes to the Colab notebook as embedded code cells. Components include Tokenizer (for mixed tabular data), MultiheadAttention, Transformer blocks, Encoder, and Decoder with Gumbel-Softmax for categorical features.

**Tech Stack:** PyTorch, Transformers architecture, Gumbel-Softmax

---

## File Structure

**Files to modify:**
- `TABCF_Colab.ipynb` - Add VAE architecture cells (~8-10 new cells)

**Files to reference:**
- `tabcf/vae/model.py` - Source VAE model classes
- Design spec: `docs/superpowers/specs/2026-06-11-tabcf-colab-migration-design.md`

---

## Task 1: Add Tokenizer Class

**Files:**
- Modify: `TABCF_Colab.ipynb` (add cells)
- Reference: `tabcf/vae/model.py:11-66`

- [ ] **Step 1: Add VAE Architecture section header**

Add markdown cell:

```markdown
## 4. VAE Model Architecture

Transformer-based Variational Autoencoder for tabular data with Gumbel-Softmax decoding.
```

- [ ] **Step 2: Add Tokenizer class header**

Add markdown cell:

```markdown
### Tokenizer

Converts mixed numerical and categorical features into token embeddings.
```

- [ ] **Step 3: Add Tokenizer class implementation**

Add code cell (adapt from `tabcf/vae/model.py:11-66`):

```python
class Tokenizer(nn.Module):
    """Tokenize mixed numerical and categorical features."""
    
    def __init__(self, d_numerical, categories, d_token, bias):
        super().__init__()
        if categories is None:
            d_bias = d_numerical
            self.category_offsets = None
            self.category_embeddings = None
        else:
            d_bias = d_numerical + len(categories)
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_token)
            nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))

        # take [CLS] token into account
        self.weight = nn.Parameter(torch.Tensor(d_numerical + 1, d_token))
        self.bias = nn.Parameter(torch.Tensor(d_bias, d_token)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    @property
    def n_tokens(self):
        return len(self.weight) + (
            0 if self.category_offsets is None else len(self.category_offsets)
        )

    def forward(self, x_num, x_cat):
        x_some = x_num if x_cat is None else x_cat
        assert x_some is not None
        x_num = torch.cat(
            [torch.ones(len(x_some), 1, device=x_some.device)]  # [CLS]
            + ([] if x_num is None else [x_num]),
            dim=1,
        )
    
        x = self.weight[None] * x_num[:, :, None]

        if x_cat is not None:
            x = torch.cat(
                [x, self.category_embeddings(x_cat + self.category_offsets[None])],
                dim=1,
            )
        if self.bias is not None:
            bias = torch.cat(
                [
                    torch.zeros(1, self.bias.shape[1], device=x.device),
                    self.bias,
                ]
            )
            x = x + bias[None]

        return x

print("✓ Tokenizer class defined")
```

- [ ] **Step 4: Test Tokenizer class**

Add code cell:

```python
# Test Tokenizer
import math

# Test with numerical features only
d_numerical = 5
d_token = 4
tokenizer = Tokenizer(d_numerical=d_numerical, categories=None, d_token=d_token, bias=True)

# Create dummy input
batch_size = 2
x_num = torch.randn(batch_size, d_numerical)
x_cat = None

# Forward pass
tokens = tokenizer(x_num, x_cat)
print(f"Input shape: {x_num.shape}")
print(f"Output shape: {tokens.shape}")
print(f"Expected: ({batch_size}, {d_numerical + 1}, {d_token})")
assert tokens.shape == (batch_size, d_numerical + 1, d_token), "Tokenizer output shape mismatch"
print("✓ Tokenizer test passed")
```

- [ ] **Step 5: Skip git commit**

(No commit - as requested)

---

## Task 2: Add MultiheadAttention Class

**Files:**
- Modify: `TABCF_Colab.ipynb` (add cells)
- Reference: `tabcf/vae/model.py:86-155`

- [ ] **Step 1: Add MultiheadAttention header**

Add markdown cell:

```markdown
### MultiheadAttention

Self-attention mechanism for transformer layers.
```

- [ ] **Step 2: Add MultiheadAttention class**

Add code cell (adapt from `tabcf/vae/model.py:86-155`):

```python
class MultiheadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, d, n_heads, dropout, initialization='kaiming'):
        if n_heads > 1:
            assert d % n_heads == 0
        assert initialization in ['xavier', 'kaiming']

        super().__init__()
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_out = nn.Linear(d, d) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        for m in [self.W_q, self.W_k, self.W_v]:
            if initialization == 'xavier' and (n_heads > 1 or m is not self.W_v):
                nn.init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            nn.init.zeros_(m.bias)
        if self.W_out is not None:
            nn.init.zeros_(self.W_out.bias)

    def _reshape(self, x):
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )

    def forward(self, x_q, x_kv, key_compression=None, value_compression=None):
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0
        if key_compression is not None:
            assert value_compression is not None
            k = key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = value_compression(v.transpose(1, 2)).transpose(1, 2)
        else:
            assert value_compression is None

        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)
        attention = F.softmax(q @ k.transpose(1, 2) / math.sqrt(d_head_key), dim=-1)
        
        if self.dropout is not None:
            attention = self.dropout(attention)
        x = attention @ self._reshape(v)
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)
        return x

print("✓ MultiheadAttention class defined")
```

- [ ] **Step 3: Test MultiheadAttention**

Add code cell:

```python
# Test MultiheadAttention
d_model = 16
n_heads = 1
batch_size = 2
seq_len = 5

attn = MultiheadAttention(d=d_model, n_heads=n_heads, dropout=0.1)
x = torch.randn(batch_size, seq_len, d_model)

output = attn(x, x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
assert output.shape == x.shape, "Attention output shape mismatch"
print("✓ MultiheadAttention test passed")
```

- [ ] **Step 4: Skip git commit**

(No commit - as requested)

---

## Task 3: Add Transformer and MLP Building Blocks

**Files:**
- Modify: `TABCF_Colab.ipynb` (add cells)
- Reference: `tabcf/vae/model.py:68-84,157-220`

- [ ] **Step 1: Add MLP class header**

Add markdown cell:

```markdown
### MLP and Transformer Blocks

Basic building blocks for the transformer encoder/decoder.
```

- [ ] **Step 2: Add MLP class**

Add code cell (from `tabcf/vae/model.py:68-84`):

```python
class MLP(nn.Module):
    """Multi-layer perceptron."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

print("✓ MLP class defined")
```

- [ ] **Step 3: Add Transformer block class**

Add code cell (adapt from `tabcf/vae/model.py:157-220`):

```python
class TransformerLayer(nn.Module):
    """Single transformer layer with self-attention and feed-forward."""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiheadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Self-attention with residual
        attn_out = self.attention(x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x

print("✓ TransformerLayer class defined")
```

- [ ] **Step 4: Test building blocks**

Add code cell:

```python
# Test MLP
mlp = MLP(input_dim=10, hidden_dim=20, output_dim=5, dropout=0.5)
x = torch.randn(2, 10)
out = mlp(x)
assert out.shape == (2, 5), "MLP output shape mismatch"
print("✓ MLP test passed")

# Test TransformerLayer
d_model = 16
transformer_layer = TransformerLayer(d_model=d_model, n_heads=1, d_ff=32, dropout=0.1)
x = torch.randn(2, 5, d_model)
out = transformer_layer(x)
assert out.shape == x.shape, "TransformerLayer output shape mismatch"
print("✓ TransformerLayer test passed")
```

- [ ] **Step 5: Skip git commit**

(No commit - as requested)

---

## Task 4: Add Encoder Model

**Files:**
- Modify: `TABCF_Colab.ipynb` (add cells)
- Reference: `tabcf/vae/model.py` (Encoder_model class)

- [ ] **Step 1: Add Encoder header**

Add markdown cell:

```markdown
### VAE Encoder

Transformer-based encoder that maps input tokens to latent space (mean and log-variance).
```

- [ ] **Step 2: Add Encoder class**

Add code cell:

```python
class Encoder_model(nn.Module):
    """VAE Encoder: tokens → latent distribution (mu, logvar)."""
    
    def __init__(self, num_tokens, d_token, n_layers, d_model, n_heads, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Project tokens to model dimension
        self.token_projection = nn.Linear(d_token, d_model)
        
        # Transformer layers
        d_ff = d_model * 4
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Latent projection (output mu and logvar)
        self.to_latent = nn.Linear(d_model, d_model * 2)
        
    def forward(self, tokens):
        # tokens: (batch, num_tokens, d_token)
        x = self.token_projection(tokens)  # (batch, num_tokens, d_model)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Use [CLS] token (first token) for latent representation
        cls_token = x[:, 0, :]  # (batch, d_model)
        
        # Project to latent space
        latent = self.to_latent(cls_token)  # (batch, d_model * 2)
        
        # Split into mu and logvar
        mu = latent[:, :self.d_model]
        logvar = latent[:, self.d_model:]
        
        return mu, logvar

print("✓ Encoder_model class defined")
```

- [ ] **Step 3: Test Encoder**

Add code cell:

```python
# Test Encoder
num_tokens = 10
d_token = 4
d_model = 16
n_layers = 2
n_heads = 1
batch_size = 2

encoder = Encoder_model(num_tokens, d_token, n_layers, d_model, n_heads)
tokens = torch.randn(batch_size, num_tokens, d_token)

mu, logvar = encoder(tokens)
print(f"Input tokens shape: {tokens.shape}")
print(f"Mu shape: {mu.shape}")
print(f"Logvar shape: {logvar.shape}")
assert mu.shape == (batch_size, d_model), "Encoder mu shape mismatch"
assert logvar.shape == (batch_size, d_model), "Encoder logvar shape mismatch"
print("✓ Encoder test passed")
```

- [ ] **Step 4: Skip git commit**

(No commit - as requested)

---

## Task 5: Add Decoder Model with Gumbel-Softmax

**Files:**
- Modify: `TABCF_Colab.ipynb` (add cells)
- Reference: `tabcf/vae/model.py` (Decoder_model class)

- [ ] **Step 1: Add Decoder header**

Add markdown cell:

```markdown
### VAE Decoder

Transformer-based decoder that reconstructs numerical and categorical features from latent space.
Uses Gumbel-Softmax for differentiable categorical reconstruction.
```

- [ ] **Step 2: Add Decoder class**

Add code cell:

```python
class Decoder_model(nn.Module):
    """VAE Decoder: latent → reconstructed features (numerical + categorical)."""
    
    def __init__(self, d_latent, num_tokens, d_token, n_layers, n_heads, 
                 d_numerical, categories, gumbel_softmax=True, tau=1.0):
        super().__init__()
        
        self.d_token = d_token
        self.d_numerical = d_numerical
        self.categories = categories
        self.gumbel_softmax = gumbel_softmax
        self.tau = tau
        
        # Project latent to token sequence
        self.latent_to_tokens = nn.Linear(d_latent, num_tokens * d_token)
        self.num_tokens = num_tokens
        
        # Transformer layers
        d_ff = d_token * 4
        self.layers = nn.ModuleList([
            TransformerLayer(d_token, n_heads, d_ff, dropout=0.1)
            for _ in range(n_layers)
        ])
        
        # Output heads
        self.to_numerical = nn.Linear(d_token, 1)  # One value per numerical feature
        
        if categories is not None:
            self.to_categorical = nn.ModuleList([
                nn.Linear(d_token, cat_size) for cat_size in categories
            ])
        else:
            self.to_categorical = None
            
    def forward(self, z):
        # z: (batch, d_latent)
        batch_size = z.shape[0]
        
        # Project to token sequence
        x = self.latent_to_tokens(z)  # (batch, num_tokens * d_token)
        x = x.view(batch_size, self.num_tokens, self.d_token)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Reconstruct numerical features (skip CLS token)
        num_tokens = x[:, 1:self.d_numerical+1, :]  # (batch, d_numerical, d_token)
        x_num = self.to_numerical(num_tokens).squeeze(-1)  # (batch, d_numerical)
        
        # Reconstruct categorical features
        if self.to_categorical is not None:
            cat_start = self.d_numerical + 1
            x_cat = []
            for i, linear in enumerate(self.to_categorical):
                cat_token = x[:, cat_start + i, :]  # (batch, d_token)
                logits = linear(cat_token)  # (batch, cat_size)
                
                if self.gumbel_softmax and self.training:
                    # Gumbel-Softmax during training
                    cat_probs = F.gumbel_softmax(logits, tau=self.tau, hard=False)
                else:
                    # Softmax during inference
                    cat_probs = F.softmax(logits, dim=-1)
                    
                x_cat.append(cat_probs)
        else:
            x_cat = None
            
        return x_num, x_cat

print("✓ Decoder_model class defined")
```

- [ ] **Step 3: Test Decoder**

Add code cell:

```python
# Test Decoder
d_latent = 16
num_tokens = 10
d_token = 4
n_layers = 2
n_heads = 1
d_numerical = 5
categories = [3, 4, 2]  # 3 categorical features with different cardinalities
batch_size = 2

decoder = Decoder_model(d_latent, num_tokens, d_token, n_layers, n_heads,
                       d_numerical, categories, gumbel_softmax=True, tau=1.0)
z = torch.randn(batch_size, d_latent)

x_num, x_cat = decoder(z)
print(f"Latent shape: {z.shape}")
print(f"Reconstructed numerical shape: {x_num.shape}")
print(f"Reconstructed categorical shapes: {[x.shape for x in x_cat]}")
assert x_num.shape == (batch_size, d_numerical), "Decoder numerical output shape mismatch"
assert len(x_cat) == len(categories), "Decoder categorical output count mismatch"
for i, cat_size in enumerate(categories):
    assert x_cat[i].shape == (batch_size, cat_size), f"Decoder categorical output {i} shape mismatch"
print("✓ Decoder test passed")
```

- [ ] **Step 4: Skip git commit**

(No commit - as requested)

---

## Task 6: Add Complete VAE Model

**Files:**
- Modify: `TABCF_Colab.ipynb` (add cells)
- Reference: `tabcf/vae/model.py` (Model_VAE class)

- [ ] **Step 1: Add VAE wrapper header**

Add markdown cell:

```markdown
### Complete VAE Model

Combines Tokenizer, Encoder, and Decoder into a full VAE with reparameterization.
```

- [ ] **Step 2: Add reparameterization function**

Add code cell:

```python
def reparameterize(mu, logvar):
    """Reparameterization trick: z = mu + std * epsilon."""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

print("✓ Reparameterization function defined")
```

- [ ] **Step 3: Add Model_VAE class**

Add code cell:

```python
class Model_VAE(nn.Module):
    """Complete Transformer-based VAE for tabular data."""
    
    def __init__(self, d_numerical, categories, config):
        super().__init__()
        
        d_token = config.d_token
        n_layers = config.num_layers
        n_heads = config.n_head
        factor = config.factor
        
        # Calculate model dimension
        d_model = d_token * factor
        
        # Tokenizer
        self.tokenizer = Tokenizer(d_numerical, categories, d_token, bias=True)
        num_tokens = self.tokenizer.n_tokens
        
        # Encoder
        self.encoder = Encoder_model(num_tokens, d_token, n_layers, d_model, n_heads)
        
        # Decoder
        self.decoder = Decoder_model(
            d_latent=d_model,
            num_tokens=num_tokens,
            d_token=d_token,
            n_layers=n_layers,
            n_heads=n_heads,
            d_numerical=d_numerical,
            categories=categories,
            gumbel_softmax=config.gumbel_softmax,
            tau=config.tau
        )
        
        self.d_model = d_model
        
    def encode(self, x_num, x_cat):
        """Encode to latent distribution."""
        tokens = self.tokenizer(x_num, x_cat)
        mu, logvar = self.encoder(tokens)
        return mu, logvar
    
    def decode(self, z):
        """Decode from latent space."""
        return self.decoder(z)
    
    def forward(self, x_num, x_cat, return_latent=False):
        """Full forward pass: encode → reparameterize → decode."""
        # Encode
        mu, logvar = self.encode(x_num, x_cat)
        
        # Reparameterize
        z = reparameterize(mu, logvar)
        
        # Decode
        x_num_recon, x_cat_recon = self.decode(z)
        
        if return_latent:
            return x_num_recon, x_cat_recon, mu, logvar, z
        else:
            return x_num_recon, x_cat_recon, mu, logvar

print("✓ Model_VAE class defined")
```

- [ ] **Step 4: Test complete VAE**

Add code cell:

```python
# Test complete VAE
d_numerical = 5
categories = [3, 4, 2]
batch_size = 2

# Use actual config from earlier
vae = Model_VAE(d_numerical, categories, config)

# Create dummy input
x_num = torch.randn(batch_size, d_numerical)
x_cat = torch.randint(0, 3, (batch_size, len(categories)))

# Forward pass
x_num_recon, x_cat_recon, mu, logvar = vae(x_num, x_cat)

print(f"Input numerical shape: {x_num.shape}")
print(f"Input categorical shape: {x_cat.shape}")
print(f"Reconstructed numerical shape: {x_num_recon.shape}")
print(f"Reconstructed categorical count: {len(x_cat_recon)}")
print(f"Latent mu shape: {mu.shape}")
print(f"Latent logvar shape: {logvar.shape}")

assert x_num_recon.shape == x_num.shape, "VAE numerical reconstruction shape mismatch"
assert len(x_cat_recon) == len(categories), "VAE categorical reconstruction count mismatch"
print("✓ Complete VAE test passed")
```

- [ ] **Step 5: Skip git commit**

(No commit - as requested)

---

## Task 7: Add Phase 2 Completion Summary

**Files:**
- Modify: `TABCF_Colab.ipynb` (add cell)

- [ ] **Step 1: Add Phase 2 completion cell**

Add markdown cell:

```markdown
---

## Phase 2 Complete ✓

**What we have:**
- ✓ Tokenizer for mixed tabular data
- ✓ MultiheadAttention mechanism
- ✓ Transformer layers (attention + feed-forward)
- ✓ VAE Encoder (tokens → latent distribution)
- ✓ VAE Decoder with Gumbel-Softmax
- ✓ Complete Model_VAE architecture

**Next Phase:** VAE training loop with loss computation and checkpointing

---
```

- [ ] **Step 2: Skip git commit**

(No commit - as requested)

---

## Self-Review Checklist

**Spec coverage:**
- ✓ Tokenizer for mixed data types
- ✓ MultiheadAttention for transformer
- ✓ Encoder architecture (transformer-based)
- ✓ Decoder with Gumbel-Softmax
- ✓ Complete VAE model wrapper
- ✓ Reparameterization trick

**No placeholders:**
- ✓ All classes have complete implementations
- ✓ All parameters are specified
- ✓ All forward methods are implemented
- ✓ Test cells verify functionality

**Type consistency:**
- ✓ All tensor shapes documented in tests
- ✓ Config parameter usage matches Phase 1
- ✓ Model dimensions are consistent

**Completeness:**
- ✓ Each component tested independently
- ✓ Full VAE integration test
- ✓ All code adapted from working source

---

## Next Steps

After Phase 2 completion:
1. **Phase 3:** VAE training loop (loss computation, optimization, checkpointing)
2. **Phase 4:** Black-box classifier training
3. **Phase 5:** Counterfactual generation
4. **Phase 6:** Evaluation and visualization
5. **Phase 7:** End-to-end integration
