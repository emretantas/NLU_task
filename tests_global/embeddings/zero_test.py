# This checks both failure modes:
# - **Zero-length sequences** (the mask issue)
# - **NaN/inf in the raw embeddings** (could cause NaN even with valid lengths)
# Can cause NaN/inf Logits Debug Error.

from pathlib import Path

import numpy as np
import torch

d_train = np.load(Path("output/train_embeddings.npz"))
d_dev = np.load(Path("output/dev_embeddings.npz"))


def run_test(d):
    prem_lengths = torch.tensor(d["premise_mask"], dtype=torch.long).sum(dim=-1)
    hyp_lengths = torch.tensor(d["hypothesis_mask"], dtype=torch.long).sum(dim=-1)

    print(f"prem zero-length: {(prem_lengths == 0).sum().item()}")
    print(f"hyp  zero-length: {(hyp_lengths == 0).sum().item()}")

    # Also check for NaN/inf in the embeddings themselves
    prem_emb = torch.tensor(d["premise_emb"], dtype=torch.float32)
    hyp_emb = torch.tensor(d["hypothesis_emb"], dtype=torch.float32)

    print(
        f"prem_emb NaN: {torch.isnan(prem_emb).any().item()}, inf: {torch.isinf(prem_emb).any().item()}"
    )
    print(
        f"hyp_emb  NaN: {torch.isnan(hyp_emb).any().item()}, inf: {torch.isinf(hyp_emb).any().item()}"
    )

    return


if __name__ == "__main__":
    print("============== TRAIN ==============")
    run_test(d_train)
    print("=============== DEV ===============")
    run_test(d_dev)
