"""
Test script for res-ESIM training with random synthetic data.

This script:
1. Generates random input data matching the expected format
2. Initializes the OracleNet model
3. Trains for one epoch using the training loop from trainer/training.py
4. Prints training metrics

NOTE: Before running this script, the following bugs must be fixed:
- res_esim/model_layers/esim_block.py:
  - Line 12: Change 'self.shared_bilstm' to 'self.bilstm'
  - Line 33: Change '_soft_dot_attention' to '_cross_attention'
  - Line 78: Remove 'self' from @staticmethod _enhance signature
- res_esim/model_layers/res_esim_block.py:
  - Line 4: Change 'from esim_block' to 'from .esim_block'
- res_esim/model_layers/stock_classifier.py:
  - Lines 44, 52: Change 'hidden_dim' parameter to 'hidden_size' in LSTM
- res_esim/model_layers/oracle_net.py:
  - Line 11: Change 'stock_classifer' to 'stock_classifier'
  - Line 45: Change 'self.classifer' to 'self.classifier'
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Import from parent module (res_esim)
from model_layers.oracle_net import OracleNet
from trainer.training import train_epoch, get_warmup_decay_scheduler
from trainer.evaluation import evaluate


# ── Synthetic Dataset ────────────────────────────────────────────────────────
class SyntheticNLIDataset(Dataset):
    """
    Generates random synthetic data for NLI task.

    Returns batches with:
    - premise_embedding: (batch, len_p, input_dim)
    - hyp_embedding: (batch, len_h, input_dim)
    - premise_length: (batch,)
    - hyp_length: (batch,)
    - label: (batch,) - 0=entailment, 1=neutral, 2=contradiction
    """

    def __init__(
        self,
        num_samples=100,
        input_dim=768,  # BERT-base dimension
        max_len=50,
        num_classes=3
    ):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.max_len = max_len
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Random sequence lengths
        premise_len = torch.randint(10, self.max_len, (1,)).item()
        hyp_len = torch.randint(10, self.max_len, (1,)).item()

        # Generate random embeddings (padded to max_len)
        premise_emb = torch.randn(self.max_len, self.input_dim)
        hyp_emb = torch.randn(self.max_len, self.input_dim)

        # Zero out padding positions
        premise_emb[premise_len:] = 0
        hyp_emb[hyp_len:] = 0

        # Random label
        label = torch.randint(0, self.num_classes, (1,)).item()

        return {
            'premise_embedding': premise_emb,
            'hyp_embedding': hyp_emb,
            'premise_length': torch.tensor(premise_len, dtype=torch.long),
            'hyp_length': torch.tensor(hyp_len, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }


# ── Main Test Function ───────────────────────────────────────────────────────
def test_single_epoch_training():
    """
    Test training for one epoch with synthetic data.
    """
    print("=" * 80)
    print("RES-ESIM SINGLE EPOCH TRAINING TEST")
    print("=" * 80)

    # ── Hyperparameters ──────────────────────────────────────────────────────
    INPUT_DIM = 768        # BERT-base embedding dimension
    HIDDEN_DIM = 300       # As per paper
    NUM_BLOCKS = 2         # Number of stacked ESIM blocks
    NUM_CLASSES = 3        # entailment, neutral, contradiction
    DROPOUT_RATE = 0.2     # As per paper (SNLI)

    BATCH_SIZE = 8         # Small batch for testing
    NUM_SAMPLES = 32       # Small dataset for testing
    MAX_LEN = 50           # Maximum sequence length

    LEARNING_RATE = 1e-4
    WARMUP_STEPS = 5
    TOTAL_STEPS = NUM_SAMPLES // BATCH_SIZE  # Steps per epoch

    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[INFO] Using device: {device}")

    # ── Create Dataset & DataLoader ──────────────────────────────────────────
    print(f"[INFO] Creating synthetic dataset...")
    print(f"       - Samples: {NUM_SAMPLES}")
    print(f"       - Batch size: {BATCH_SIZE}")
    print(f"       - Input dim: {INPUT_DIM}")
    print(f"       - Max length: {MAX_LEN}")

    dataset = SyntheticNLIDataset(
        num_samples=NUM_SAMPLES,
        input_dim=INPUT_DIM,
        max_len=MAX_LEN,
        num_classes=NUM_CLASSES
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    # ── Initialize Model ─────────────────────────────────────────────────────
    print(f"\n[INFO] Initializing OracleNet model...")
    print(f"       - Hidden dim: {HIDDEN_DIM}")
    print(f"       - Num blocks: {NUM_BLOCKS}")
    print(f"       - Num classes: {NUM_CLASSES}")
    print(f"       - Dropout rate: {DROPOUT_RATE}")

    try:
        model = OracleNet(
            input_dim=INPUT_DIM,
            hidden_dim=HIDDEN_DIM,
            num_blocks=NUM_BLOCKS,
            num_classes=NUM_CLASSES,
            dropout_rate=DROPOUT_RATE
        ).to(device)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"       - Total parameters: {total_params:,}")
        print(f"       - Trainable parameters: {trainable_params:,}")

    except Exception as e:
        print(f"\n[ERROR] Failed to initialize model!")
        print(f"        {type(e).__name__}: {e}")
        print("\n[HINT] Please fix the bugs listed at the top of this script first.")
        return

    # ── Setup Training Components ────────────────────────────────────────────
    print(f"\n[INFO] Setting up training components...")

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )

    # Learning rate scheduler (warmup + linear decay)
    scheduler = get_warmup_decay_scheduler(
        optimizer,
        warmup_steps=WARMUP_STEPS,
        total_steps=TOTAL_STEPS
    )

    # Loss function
    criterion = nn.CrossEntropyLoss()

    print(f"       - Optimizer: Adam (lr={LEARNING_RATE}, weight_decay=0.01)")
    print(f"       - Scheduler: Warmup({WARMUP_STEPS}) + Linear Decay")
    print(f"       - Loss: CrossEntropyLoss")

    # ── Train One Epoch ──────────────────────────────────────────────────────
    print(f"\n[INFO] Starting training for 1 epoch...")
    print(f"       - Total steps: {TOTAL_STEPS}")
    print("-" * 80)

    try:
        avg_loss, accuracy, macro_f1 = train_epoch(
            oracle=model,
            loader=loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device
        )

        # ── Print Results ────────────────────────────────────────────────────
        print("-" * 80)
        print(f"\n[RESULTS] Training completed successfully!")
        print(f"          - Average Loss: {avg_loss:.4f}")
        print(f"          - Accuracy: {accuracy:.2%}")
        print(f"          - Macro F1: {macro_f1:.4f}")

        # ── Test Evaluation ──────────────────────────────────────────────────
        print(f"\n[INFO] Testing evaluation on a dev set...")
        dev_dataset = SyntheticNLIDataset(
            num_samples=16,
            input_dim=INPUT_DIM,
            max_len=MAX_LEN,
            num_classes=NUM_CLASSES
        )
        dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE)

        dev_loss, dev_acc, dev_f1 = evaluate(
            oracle=model,
            loader=dev_loader,
            criterion=criterion,
            device=device
        )

        print(f"          - Dev Loss: {dev_loss:.4f}")
        print(f"          - Dev Accuracy: {dev_acc:.2%}")
        print(f"          - Dev Macro F1: {dev_f1:.4f}")

    except Exception as e:
        print(f"\n[ERROR] Training failed!")
        print(f"        {type(e).__name__}: {e}")
        print("\n[HINT] Please check the error trace above.")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)


# ── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_single_epoch_training()
