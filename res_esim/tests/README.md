# Tests for Res-ESIM

## Overview

This directory contains test scripts for the res-ESIM implementation.

## Files

- **test_training.py**: Tests single-epoch training with synthetic random data

## Running the Tests

### Running test_training.py

Once the bugs are fixed, run:

```bash
cd /Users/vpremakantha/Documents/UOM/Y3/NLU/NLU-CW
python res_esim/tests/test_training.py
```

or from the res_esim directory:

```bash
cd res_esim
python tests/test_training.py
```

or from the tests directory:

```bash
cd res_esim/tests
python test_training.py
```

## What the Test Does

**test_training.py**:
1. Creates a synthetic NLI dataset with random embeddings
2. Initializes the OracleNet model (ResESIM encoder + StockClassifier)
3. Sets up optimizer (Adam), scheduler (warmup + linear decay), and loss function
4. Trains for one complete epoch
5. Reports training metrics: loss, accuracy, and macro-F1

### Test Configuration

- **Input dimension**: 768 (BERT-base size)
- **Hidden dimension**: 300 (as per paper)
- **Number of ESIM blocks**: 2
- **Dataset size**: 32 samples (for quick testing)
- **Batch size**: 8
- **Sequence length**: Random between 10-50 tokens
- **Classes**: 3 (entailment, neutral, contradiction)

### Expected Output

```
================================================================================
RES-ESIM SINGLE EPOCH TRAINING TEST
================================================================================

[INFO] Using device: cpu (or cuda if available)
[INFO] Creating synthetic dataset...
       - Samples: 32
       - Batch size: 8
       - Input dim: 768
       - Max length: 50

[INFO] Initializing OracleNet model...
       - Hidden dim: 300
       - Num blocks: 2
       - Num classes: 3
       - Dropout rate: 0.2
       - Total parameters: XXX,XXX
       - Trainable parameters: XXX,XXX

[INFO] Setting up training components...
       - Optimizer: Adam (lr=0.0001, weight_decay=0.01)
       - Scheduler: Warmup(5) + Linear Decay
       - Loss: CrossEntropyLoss

[INFO] Starting training for 1 epoch...
       - Total steps: 4
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

[RESULTS] Training completed successfully!
          - Average Loss: X.XXXX
          - Accuracy: XX.XX%
          - Macro F1: X.XXXX

================================================================================
TEST COMPLETED
================================================================================
```

## Troubleshooting

If you encounter import errors:
- Ensure you're running from the project root directory
- Check that all `__init__.py` files exist in the module directories

If you encounter attribute errors:
- Double-check that all bugs listed in Prerequisites have been fixed

If CUDA out of memory:
- The test will automatically fall back to CPU
- Reduce `BATCH_SIZE` or `NUM_SAMPLES` in the script if needed
