import argparse
import json
import os
from pathlib import Path
import multiprocessing as mp

import optuna
import torch
from tqdm import tqdm

from res_esim.loader.res_esim_dataset import ResESIM_Dataset
from res_esim.model_layers.oracle_net import OracleNet
from res_esim.run.train import HyperParameters, train

# --- Configuration ------------------------------------
TRAIN_PT = Path("output/train_embeddings.npz")
DEV_PT = Path("output/dev_embeddings.npz")
HYPERTUNING_DIR = Path("output/hypertuning")
STUDY_NAME = "res_esim_hypertuning"
STORAGE_NAME = f"sqlite:///{HYPERTUNING_DIR}/{STUDY_NAME}.db"


def objective(trial, device, train_dataset, dev_dataset):
    # --- Define Search Space --------------------------
    hidden_dim = trial.suggest_categorical("HIDDEN_DIM", [256, 512, 768, 1024])
    num_blocks = trial.suggest_int("NUM_BLOCKS", 1, 6)
    dropout_rate = trial.suggest_float("DROPOUT_RATE", 0.1, 0.4, step=0.1)
    learning_rate = trial.suggest_float("LEARNING_RATE", 5e-5, 2e-4, log=True)
    num_attn_heads = trial.suggest_categorical("NUM_ATTN_HEADS", [4, 8, 12])

    # Check for valid combination (hidden_dim % num_attn_heads == 0)
    if hidden_dim % num_attn_heads != 0:
        raise optuna.exceptions.TrialPruned()

    # --- Setup HyperParameters -----------------------
    params = HyperParameters(
        HIDDEN_DIM=hidden_dim,
        NUM_BLOCKS=num_blocks,
        DROPOUT_RATE=dropout_rate,
        LEARNING_RATE=learning_rate,
        NUM_ATTN_HEADS=num_attn_heads,
        NUM_EPOCHS=15,
        BATCH_SIZE=32,
    )

    # --- Initialize Model ----------------------------
    model = OracleNet(
        input_dim=params.INPUT_DIM,
        hidden_dim=params.HIDDEN_DIM,
        num_blocks=params.NUM_BLOCKS,
        num_classes=params.NUM_CLASSES,
        dropout_rate=params.DROPOUT_RATE,
        num_heads=params.NUM_ATTN_HEADS,
    )
    model.to(device)

    # --- Run Training -------------------------------
    run_name = f"trial_{trial.number}"
    try:
        best_f1, _ = train(
            model=model,
            device=device,
            hyperparameters=params,
            train_dataset=train_dataset,
            dev_dataset=dev_dataset,
            run_name=run_name,
            base_out_dir=HYPERTUNING_DIR / "trials",
        )
    except Exception as e:
        print(f"Trial {trial.number} on {device} failed with error: {e}")
        return 0.0

    return best_f1


def run_worker(gpu_id, trials_per_worker, platform, train_dataset, dev_dataset):
    """
    A single worker process that runs a subset of the total trials on a specific GPU.
    """
    if platform == "hpc":
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    print(f"[Worker GPU:{gpu_id}] Starting {trials_per_worker} trials on {device}...")

    # Load existing study or create new one
    study = optuna.load_study(
        study_name=STUDY_NAME,
        storage=STORAGE_NAME,
    )

    study.optimize(
        lambda trial: objective(trial, device, train_dataset, dev_dataset),
        n_trials=trials_per_worker,
    )
    print(f"[Worker GPU:{gpu_id}] Finished.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", choices=["mac", "hpc"], default="mac")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--num_gpus", type=int, default=2, help="Number of GPUs to utilize")
    args = parser.parse_args()

    HYPERTUNING_DIR.mkdir(parents=True, exist_ok=True)

    # --- Initial Setup ------------------------------
    print(f"Initializing Study: {STUDY_NAME}")
    # Create the study once in the main process to ensure it exists
    optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_NAME,
        direction="maximize",
        load_if_exists=True,
    )

    # --- Load Data Once (Shared Memory) -------------
    print("Loading datasets into memory...")
    train_dataset = ResESIM_Dataset(TRAIN_PT)
    dev_dataset = ResESIM_Dataset(DEV_PT)

    # --- Multiprocessing ----------------------------
    num_workers = args.num_gpus if args.platform == "hpc" else 1
    trials_per_worker = args.trials // num_workers
    
    print(f"Spawning {num_workers} workers to run ~{trials_per_worker} trials each...")

    processes = []
    for i in range(num_workers):
        p = mp.Process(
            target=run_worker,
            args=(i, trials_per_worker, args.platform, train_dataset, dev_dataset)
        )
        p.start()
        processes.append(p)

    # Wait for all workers to finish
    for p in processes:
        p.join()

    # --- Results Summary ---------------------------
    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE_NAME)
    
    print("\n" + "=" * 30)
    print("HYPERTUNING COMPLETE")
    print(f"Best trial: {study.best_trial.number}")
    print(f"  Value (Macro F1): {study.best_value:.4f}")
    print("  Params: ")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    print("=" * 30)

    with open(HYPERTUNING_DIR / "best_hyperparameters.json", "w") as f:
        json.dump(study.best_params, f, indent=2)


if __name__ == "__main__":
    # Required for multiprocessing on some platforms
    mp.set_start_method('spawn', force=True)
    main()
