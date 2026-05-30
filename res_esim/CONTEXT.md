# Res-ESIM Implementation Context

## Project Overview
Implementation of the **Residual Connected Enhanced Sequential Inference Model** (res-ESIM) for Natural Language Inference (NLI), based on the paper by Li et al. (2019).

**Task**: Classify sentence pairs (premise, hypothesis) into three categories:
- Entailment
- Neutral
- Contradiction

---

## Architecture Overview (from Paper)

The res-ESIM model consists of 4 main components:

### 1. Embedding Layer
- **Current**: BERT-based contextual embeddings
- **Future**: Will be replaced with ELMo embeddings
- **Purpose**: Convert tokens to dense vector representations
- **Abstraction Point**: The model accepts pre-computed embeddings as input, making it agnostic to the embedding source

### 2. Residual Connected Block (N × ESIM-blocks)
Each ESIM block contains:
- **Encoding Layer** (BiLSTM): Encodes premise and hypothesis separately using a shared BiLSTM
- **Interaction Layer** (Soft Attention): Cross-sentence attention mechanism
  - Premise attends to hypothesis → h'_p
  - Hypothesis attends to premise → h'_h
- **Enhancement Layer**: Creates rich representations via concatenation
  - `m = [h, h_att, h - h_att, h * h_att]` (4 × hidden_dim)
- **FFN Layer**: Projects enhanced representation back to hidden_dim
- **Residual Connection + LayerNorm**: `output = LayerNorm(input + FFN(enhancement))`

**Key Innovation**: Stacking N blocks with residual connections prevents degradation in deep networks.

### 3. Aggregation Layer (BiLSTM)
- Separate BiLSTMs for premise and hypothesis
- Captures sequential dependencies after all ESIM blocks
- Outputs: `v_p` and `v_h`

### 4. Classification Layer
- **Max Pooling**: Extract most salient features (with masking for padding)
- **Concatenation**: `v = [v_p_max; v_h_max; v_p_max - v_h_max; v_p_max * v_h_max]`
- **FFN**: Two-layer network with ReLU activation
  - W4: (4 × hidden_dim) → hidden_dim
  - W5: hidden_dim → num_classes (3)
- **Output**: Class logits (softmax applied in loss function)

---

## Current Implementation Status

### File Structure
```
res_esim/
├── model_layers/
│   ├── esim_block.py          # Single ESIM block
│   ├── res_esim_block.py      # Stacked ESIM blocks with residual connections
│   ├── stock_classifier.py    # Aggregation + classification layers
│   └── oracle_net.py          # Unified model (encoder + classifier)
├── trainer/
│   ├── training.py            # Training loop with warmup/decay scheduler
│   ├── evaluation.py          # Evaluation metrics (loss, accuracy, F1)
│   └── inference.py           # Prediction interface
├── tests/
│   ├── test_training.py       # Single-epoch training test with synthetic data
│   └── README.md              # Test documentation
├── CONTEXT.md                 # This file
└── Residual_Connected_Enhanced_Sequential_Inference_Model_for_Natural_Language_Inference.pdf
```

### Implementation Details

#### 1. **ESIMBlock** (`esim_block.py`)
**Purpose**: Single ESIM block with attention, enhancement, and residual connection

**Status**: ✅ All naming bugs fixed, ⚠️ NaN issue present

**Fixed Issues**:
- ✅ Consistent naming: uses `self.shared_bilstm` throughout
- ✅ Method name: uses `_soft_dot_attention` throughout
- ✅ `_enhance` is correctly @staticmethod without `self` parameter

**Current Issues**:
- ⚠️ **CRITICAL: NaN in softmax** - Using `float('-inf')` for masking causes NaN when entire rows/columns are masked
  - Affects ~143/400 positions in testing (36% of attention computations)
  - First appears in `beta` (hypothesis attending to premise)
  - Propagates through entire network causing NaN loss
  - **Solution**: Replace `-inf` with large negative value (e.g., `-1e9`) or use `pack_padded_sequence`
- 📝 Debug code added at lines 69-75, 79-85 to track NaN propagation

**Expected Flow**:
```
Input: (h_p, h_h) → BiLSTM → Attention → Enhancement → FFN → Residual + LayerNorm → Output
```

#### 2. **ResESIM** (`res_esim_block.py`)
**Purpose**: Stack N ESIM blocks with residual connections

**Status**: ✅ All issues fixed

**Fixed Issues**:
- ✅ Line 4: Correct relative import `from .esim_block import ESIMBlock`

**Features**:
✅ Input projection layer (adapts input_dim to hidden_dim)
✅ Padding mask generation
✅ Sequential processing through N ESIM blocks
✅ Dropout for regularization

#### 3. **StockClassifier** (`stock_classifier.py`)
**Purpose**: Aggregation BiLSTM + max pooling + FFN classification

**Status**: ✅ LSTM params fixed, 📝 Debug code added

**Fixed Issues**:
- ✅ Lines 44, 52: Correct LSTM parameter `hidden_size`

**Debug Additions**:
- 📝 Lines 88-95: Debug checks for -inf/NaN in max pooling
- 📝 Lines 106-110: Debug checks for NaN/inf in concatenated vector
- 📝 Lines 118-120: Debug checks for NaN/inf in final logits
- 📝 Line 28: TODO comment about NaN issue

**Features**:
✅ Separate BiLSTMs for premise and hypothesis (as per paper)
✅ Masked max pooling (prevents padding tokens from affecting pooling)
✅ Enhancement concatenation `[v_p; v_h; v_p - v_h; v_p * v_h]`
✅ Two-layer FFN (W4 + ReLU + W5)
✅ Returns logits (not probabilities)

#### 4. **OracleNet** (`oracle_net.py`)
**Purpose**: Unified end-to-end model combining encoder and classifier

**Status**: ✅ All issues fixed

**Fixed Issues**:
- ✅ Lines 10-11: Correct imports with relative paths
- ✅ Line 45: Correct attribute name `self.classifier`

**Architecture**:
```python
Input (premise_emb, hyp_emb, lengths)
    ↓
ResESIM (encoder) → (h_p, h_h, mask_p, mask_h)
    ↓
StockClassifier → logits
```

**Model Stats** (with default hyperparameters):
- Total parameters: 3,484,503
- Trainable parameters: 3,484,503

#### 5. **Training Infrastructure** (`trainer/`)

**training.py**:
✅ Warmup + linear decay scheduler (as per paper)
✅ Gradient clipping (max_norm=1.0)
✅ One-epoch training function
✅ Metrics: loss, accuracy, macro-F1

**Known Issues**:
- ⚠️ **Line 74: Accuracy calculation bug** - Uses integer division (`//`) instead of float division (`/`)
  - Results in 0.00% accuracy even when predictions are correct
  - Example: 5 correct out of 32 → `5 // 32 = 0` instead of `5 / 32 = 0.15625`
  - **Fix**: Change `//` to `/` on line 74

**evaluation.py**:
✅ Evaluation on dev set
✅ Returns loss, accuracy, macro-F1
⚠️ Note: Label key in batch is `'labels'` (differs from training.py)

**inference.py**:
✅ Prediction function
✅ Label mapping (IDX2LABEL)
⚠️ Note: Label key in batch is inconsistent (`'label'` in training.py vs `'labels'` in evaluation.py)

---

## Key Design Decisions

### 1. **Embedding Abstraction**
The model accepts pre-computed embeddings as input rather than raw text:
- **Input**: `(batch, seq_len, input_dim)` embedding tensors
- **Benefit**: Decouples embedding method (BERT, ELMo, GloVe) from core architecture
- **Flexibility**: Can swap embedding sources without changing model code

### 2. **Padding Mask Handling**
- Masks are `True` for padding positions
- Applied as `-inf` before softmax (attention, max pooling)
- Prevents padding tokens from influencing attention weights or pooling results

### 3. **Residual Connections**
- Formula: `output = LayerNorm(input + transformation(input))`
- Enables gradient flow in deep networks
- Prevents degradation as network depth increases

### 4. **Shared vs. Independent BiLSTMs**
- **ESIM Blocks**: Shared BiLSTM for premise and hypothesis (parameter efficiency)
- **Aggregation Layer**: Separate BiLSTMs (allows specialization)

---

## Hyperparameters (from Paper)

- **hidden_dim**: 300 (for SNLI experiments)
- **num_blocks**: 1-6 (paper experiments with different depths)
- **dropout_rate**: 0.2 (SNLI), 0.1 (MultiNLI)
- **batch_size**: 64
- **learning_rate**: Warmup then linear decay to 0
- **gradient_clipping**: max_norm = 1.0
- **optimizer**: Adam (β1=0.9, β2=0.999, weight_decay=0.01)

---

## Data Format

Expected batch structure:
```python
{
    'premise_embedding': Tensor(batch, len_p, input_dim),
    'hyp_embedding': Tensor(batch, len_h, input_dim),
    'premise_length': Tensor(batch,),  # actual lengths (for masking)
    'hyp_length': Tensor(batch,),
    'label': Tensor(batch,)  # 0=entailment, 1=neutral, 2=contradiction
}
```

---

## Current Git Status

**Branch**: `esim/res-esim`
**Main Branch**: `main`

**Recent Commits**:
1. `1cf29c5` - Added unified model: OracleNet
2. `9f88156` - Added training loop for one step, treating classifier & encoder as different entities
3. `a5912e9` - Added Stock Classifier from ESIM Paper
4. `bd8c330` - Added evaluation and inference logic
5. `048215a` - Added RESESIM

---

## Testing Results & Known Issues

### Test Setup
- **Test script**: `tests/test_training.py`
- **Dataset**: 32 synthetic samples, batch size 8
- **Model**: OracleNet with 2 ESIM blocks, hidden_dim=300
- **Status**: Model initializes and trains, but produces NaN loss

### Test Results (with bugs)
```
Average Loss: nan
Accuracy: 18.75%  (due to integer division bug)
Macro F1: 0.1053
```

### Critical Issue: NaN in Attention Softmax

**Root Cause**:
- Using `float('-inf')` for padding mask in attention mechanism
- When entire rows/columns are masked to -inf, `softmax([-inf, -inf, ...])` → **NaN**
- Observed: ~143/400 positions (36%) have all-inf rows/columns in first batch
- NaN first appears in `beta` (hypothesis attending to premise) in first ESIM block
- Propagates through entire network

**Debug Output Analysis**:
```
[DEBUG] NaN detected in beta (hypothesis attending to premise)!
        mask_p sum per sequence: tensor([19, 34, 27, 20, 29, 32, 36, 13])
        Number of all-inf columns in e_t: 143
```

**Why This Happens**:
1. BiLSTM processes padding positions (no `pack_padded_sequence` used)
2. BiLSTM hidden states propagate even for padding tokens
3. Some BiLSTM outputs for padding become problematic values
4. Combined with aggressive -inf masking → all-inf rows/columns
5. Softmax on all-inf → NaN

**Proposed Solutions** (in priority order):
1. ✅ **Replace `-inf` with `-1e9`** (simple, immediate fix)
   - Large negative value achieves same masking effect
   - Avoids division by zero in softmax
   - No architectural changes needed

2. 🔄 **Use `pack_padded_sequence`** (better solution)
   - Properly tells BiLSTM to skip padding positions
   - More memory efficient
   - Requires changing BiLSTM forward pass in both ESIM and classifier

3. 🔄 **Add epsilon to softmax** (workaround)
   - Replace NaN values with zeros after detection
   - Less elegant, treats symptom not cause

### Secondary Issue: Accuracy Calculation

**Bug**: `trainer/training.py` line 74 uses integer division
```python
accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) // len(all_labels)
```

**Result**: Always shows 0.00% accuracy unless > 50% correct

**Fix**: Change `//` to `/`

---

## Next Steps

### Immediate Fixes Required:
1. ✅ **Fix ESIMBlock bugs** - DONE (naming, method signatures)
2. ✅ **Fix import paths** - DONE (relative imports)
3. ✅ **Fix typos** - DONE (stock_classifier import, LSTM parameters)
4. 🔧 **Fix NaN issue** - Replace `-inf` with `-1e9` in attention masking
5. 🔧 **Fix accuracy calculation** - Change `//` to `/` in training.py line 74
6. 🔧 **Standardize batch keys** (`'label'` vs `'labels'`)

### Future Work:
1. **Replace BERT with ELMo** embeddings
2. **Implement `pack_padded_sequence`** for proper padding handling
3. **Add data loading pipeline** for SNLI/MultiNLI
4. **Implement full training script** with checkpointing
5. **Hyperparameter tuning** (number of blocks, hidden dimension)
6. **Add model evaluation** on test sets
7. **Remove debug print statements** once issues are resolved

---

## Paper Reference

Li, Y., Wang, J., Lin, H., Zhang, S., & Yang, Z. (2019). *Residual Connected Enhanced Sequential Inference Model for Natural Language Inference*. IEEE.

**Key Contributions**:
- Residual connections prevent degradation in stacked ESIM blocks
- Enhanced contextual embeddings (BERT) improve performance
- Achieves competitive results on SNLI and MultiNLI datasets
