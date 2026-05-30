#!/usr/bin/env python
"""
Standalone inference script for ResESIM NLI model.
Replaces the notebook for local testing.
"""

import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from res_esim.loader.res_esim_dataset import ResESIM_Dataset
from res_esim.model_layers.oracle_net import OracleNet
from precomputeClasses import EmbeddingPrecomputer

print("=" * 70)
print("ResESIM NLI — Inference Pipeline")
print("=" * 70)

# ─── Step 1: Device Selection ────────────────────────────────────────────
print("\n[1/5] Selecting device...")
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"      Device: {device}")

# ─── Step 2: Check meta.pt ───────────────────────────────────────────────
print("\n[2/5] Verifying meta.pt...")
meta_path = './notebook_data/meta.pt'
if not Path(meta_path).exists():
    print(f"      ERROR: {meta_path} not found!")
    print("      Run: python build_meta.py")
    exit(1)
print(f"      ✓ {meta_path} exists")

# ─── Step 3: Pre-compute Embeddings ──────────────────────────────────────
print("\n[3/5] Pre-computing embeddings...")
emb_output = './notebook_data/inference_embeddings.npz'

if Path(emb_output).exists():
    print(f"      ✓ {emb_output} already exists, skipping pre-computation")
else:
    print(f"      Computing embeddings...")
    pc = EmbeddingPrecomputer(
        meta_path    = meta_path,
        elmo_options = './notebook_data/elmo_model/options.json',
        elmo_weights = './notebook_data/elmo_model/weights.hdf5',
        elmo_venv    = './notebook_data/myenv/bin/python',
    )
    pc.run(
        csv_path   = './data/NLI_trial.csv',
        output_npz = emb_output,
    )
    print(f"      ✓ Saved {emb_output}")

# ─── Step 4: Load Model + Data ───────────────────────────────────────────
print("\n[4/5] Loading model and data...")

# Load dataset
dataset = ResESIM_Dataset(emb_output)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
print(f"      ✓ Loaded dataset: {len(dataset)} samples")

# Load model
model = OracleNet(
    input_dim=1477,
    hidden_dim=512,
    num_blocks=3,
    num_classes=2,
    dropout_rate=0.2,
    num_heads=8,
)
model.load_state_dict(torch.load('./final_model_versions/ff2f02d4/best_model.pt', map_location=device))
model.to(device).eval()
print(f"      ✓ Model loaded: OracleNet (ff2f02d4)")

# ─── Step 5: Run Inference ───────────────────────────────────────────────
print("\n[5/5] Running inference...")
all_preds = []
with torch.no_grad():
    for batch in tqdm(loader, desc='inference'):
        logits = model(
            batch['premise_embedding'].to(device),
            batch['hyp_embedding'].to(device),
            batch['premise_length'].to(device),
            batch['hyp_length'].to(device),
        )
        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())

print(f"      ✓ Inference complete: {len(all_preds)} predictions")

# ─── Save Results ────────────────────────────────────────────────────────
print("\n[SAVE] Saving predictions...")
test_df = pd.read_csv('./data/NLI_trial.csv')
test_df['label'] = all_preds
test_df[['premise', 'hypothesis', 'label']].to_csv('Group_n_B.csv', index=False)

print(f"      ✓ Saved Group_n_B.csv ({len(all_preds)} predictions)")
print(f"      Label distribution: {pd.Series(all_preds).value_counts().sort_index().to_dict()}")

print("\n" + "=" * 70)
print("✓ Done. Submit Group_n_B.csv via Canvas.")
print("=" * 70)
