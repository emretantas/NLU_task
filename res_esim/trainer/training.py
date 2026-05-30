from torch.optim.lr_scheduler import LinearLR, SequentialLR
import torch.nn as nn

from sklearn.metrics import f1_score
from tqdm import tqdm

# --- LR Scheduler -------------------------------------
def get_warmup_decay_scheduler(optimizer, warmup_steps, total_steps):
    # Linear warmup, then decay to 0.
    # (Li et al. §III-B)

    warmup = LinearLR(
        optimizer,
        start_factor    = 1 / warmup_steps,
        end_factor      = 1.0,
        total_iters     = warmup_steps
    )

    decay = LinearLR(
        optimizer,
        start_factor    = 1.0,
        end_factor      = 0.0,
        total_iters     = total_steps - warmup_steps
    )

    return SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[warmup_steps])


# --- Training Step (One Epoch) --------------------------
def train_epoch(oracle, loader, optimizer, scheduler, criterion, device, epoch=None):
    # Runs one full pass over training set, on unified OracleNet.

    oracle.train()

    total_loss = 0.0
    all_preds, all_labels = [], []

    desc = f"Epoch {epoch}" if epoch is not None else "Batch"
    batch_bar = tqdm(loader, desc=desc, unit="batch", leave=False)
    for batch in batch_bar:
        # --- Unpack & Move to Device ------------------
        premise_embedding = batch['premise_embedding'].to(device)   # (batch, len_p, input_dim)
        hyp_embedding     = batch['hyp_embedding'].to(device)       # (batch, len
        premise_length    = batch['premise_length'].to(device)      # (batch,)
        hyp_length        = batch['hyp_length'].to(device)          # (batch,)

        labels            = batch['label'].to(device)

        optimizer.zero_grad()

        # --- Forward Pass on Oracle ------------------
        logits = oracle(
            premise_embedding, hyp_embedding, premise_length, hyp_length
        )
        loss = criterion(logits, labels)

        # --- Back Propogation -------------------------
        loss.backward()

        ## Gradient Clipping
        nn.utils.clip_grad_norm_(
            oracle.parameters(),
            max_norm=1.0
        )

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        batch_bar.set_postfix(loss=f"{loss.item():.4f}")
        preds = logits.argmax(dim=-1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    # --- Training Loss Calculation ---------------------
    avg_loss = total_loss / len(loader)
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, accuracy, macro_f1
