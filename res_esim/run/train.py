import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import res_esim.trainer.evaluation as res_esim_eval
import res_esim.trainer.training as res_esim_trainer
from res_esim.loader.res_esim_dataset import ResESIM_Dataset
from res_esim.model_layers.oracle_net import OracleNet


class HyperParameters:
    def __init__(self, **kwargs):
        self.INPUT_DIM = kwargs.get("INPUT_DIM", 1477)
        self.HIDDEN_DIM = kwargs.get("HIDDEN_DIM", 512)
        self.NUM_BLOCKS = kwargs.get("NUM_BLOCKS", 3)
        self.NUM_CLASSES = kwargs.get("NUM_CLASSES", 2)
        self.DROPOUT_RATE = kwargs.get("DROPOUT_RATE", 0.2)
        self.NUM_ATTN_HEADS = kwargs.get("NUM_ATTN_HEADS", 8)
        self.NUM_EPOCHS = kwargs.get("NUM_EPOCHS", 20)
        self.BATCH_SIZE = kwargs.get("BATCH_SIZE", 32)
        self.LEARNING_RATE = kwargs.get("LEARNING_RATE", 1e-4)

        # These will be calculated
        self.NUM_SAMPLES = 0
        self.TOTAL_STEPS = 0
        self.WARMUP_STEPS = 0


def train(
    model: OracleNet,
    device,
    hyperparameters: HyperParameters,
    train_dataset=None,
    dev_dataset=None,
    run_name=None,
    base_out_dir=None,
):
    # --- Setup Data Loaders ------------------------------
    TRAIN_PT = Path("output/train_embeddings.npz")
    DEV_PT = Path("output/dev_embeddings.npz")

    if train_dataset is None:
        train_dataset = ResESIM_Dataset(TRAIN_PT)
    if dev_dataset is None:
        dev_dataset = ResESIM_Dataset(DEV_PT)

    train_loader = DataLoader(
        train_dataset,
        batch_size=hyperparameters.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=hyperparameters.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # --- Update Hyper Parameters -------------------------
    hyperparameters.NUM_SAMPLES = len(train_dataset)
    hyperparameters.TOTAL_STEPS = (
        hyperparameters.NUM_SAMPLES // hyperparameters.BATCH_SIZE
    ) * hyperparameters.NUM_EPOCHS
    hyperparameters.WARMUP_STEPS = int(hyperparameters.TOTAL_STEPS * 0.05)

    # --- Setup Training Components -----------------------
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hyperparameters.LEARNING_RATE,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    # Scheduler
    scheduler = res_esim_trainer.get_warmup_decay_scheduler(
        optimizer=optimizer,
        warmup_steps=hyperparameters.WARMUP_STEPS,
        total_steps=hyperparameters.TOTAL_STEPS,
    )

    # Loss
    criterion = nn.CrossEntropyLoss()

    # --- Output Directory ---------------------------------
    if run_name is None:
        run_name = hashlib.sha256(os.urandom(16)).hexdigest()[:8]

    if base_out_dir is None:
        base_out_dir = Path("output/training_runs")

    out_dir = Path(base_out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Training Loop ------------------------------------
    best_loss = float("inf")
    best_f1 = float("-inf")

    history = {"train_loss": [], "dev_f1": []}

    epoch_bar = tqdm(
        range(hyperparameters.NUM_EPOCHS),
        desc=f"Run {run_name}",
        unit="epoch",
        leave=False,
    )
    for epoch in epoch_bar:
        train_loss, train_acc, train_f1 = res_esim_trainer.train_epoch(
            oracle=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            epoch=epoch,
        )

        dev_loss, dev_acc, dev_f1 = res_esim_eval.evaluate(
            oracle=model, loader=dev_loader, criterion=criterion, device=device
        )

        epoch_bar.set_postfix(t_f1=f"{train_f1:.4f}", d_f1=f"{dev_f1:.4f}")
        history["train_loss"].append(train_loss)
        history["dev_f1"].append(dev_f1)

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_loss = dev_loss
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            with open(out_dir / "meta.json", "w") as f:
                json.dump(
                    {
                        "train_pt": str(TRAIN_PT),
                        "dev_pt": str(DEV_PT),
                        "trained_at": datetime.now().isoformat(),
                        "best_epoch": epoch,
                        "best_train_loss": train_loss,
                        "best_train_acc": train_acc,
                        "best_train_f1": train_f1,
                        "best_dev_loss": dev_loss,
                        "best_dev_acc": dev_acc,
                        "best_dev_f1": dev_f1,
                        "hyperparameters": {
                            k: v for k, v in vars(hyperparameters).items()
                        },
                    },
                    f,
                    indent=2,
                )

    # --- Plot Training Curves -----------------------------
    epochs = range(1, hyperparameters.NUM_EPOCHS + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], marker="o")
    ax1.set_title("Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    ax2.plot(epochs, history["dev_f1"], marker="o", color="orange")
    ax2.set_title("Dev Macro F1")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1")

    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close(fig)

    return best_f1, out_dir


def initialize_and_train(platform: str = "mac"):
    hyper_parameters = HyperParameters()
    model = OracleNet(
        input_dim=hyper_parameters.INPUT_DIM,
        hidden_dim=hyper_parameters.HIDDEN_DIM,
        num_blocks=hyper_parameters.NUM_BLOCKS,
        num_classes=hyper_parameters.NUM_CLASSES,
        dropout_rate=hyper_parameters.DROPOUT_RATE,
        num_heads=hyper_parameters.NUM_ATTN_HEADS,
    )

    if platform == "hpc":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on {device}")

    model.to(device)

    print("starting training....")
    best_f1, out_dir = train(
        model=model, device=device, hyperparameters=hyper_parameters
    )

    print(f"Training Complete, Best F1: {best_f1}, saved to {out_dir}")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", choices=["mac", "hpc"], default="mac")
    args = parser.parse_args()
    initialize_and_train(platform=args.platform)
