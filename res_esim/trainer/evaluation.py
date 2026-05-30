import torch

from sklearn.metrics import f1_score

def evaluate(oracle, loader, criterion, device):
    """
    Evaluate on dev set.
    Returns loss, accuracy, macro-F1.
    """

    oracle.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            # --- Unpack & Move to Device ------------------
            premise_embedding = batch['premise_embedding'].to(device)   # (batch, len_p, input_dim)
            hyp_embedding     = batch['hyp_embedding'].to(device)       # (batch, len_h, input_dim)
            premise_length    = batch['premise_length'].to(device)      # (batch,)
            hyp_length        = batch['hyp_length'].to(device)          # (batch,)

            label             = batch['label'].to(device)              # (batch,)

            # --- Forward Pass -----------------------------
            logits = oracle(
                premise_embedding, hyp_embedding, premise_length, hyp_length
            )
            loss = criterion(logits, label)

            total_loss += loss.item() * label.size(0)
            preds = torch.argmax(logits, dim=-1)  # (batch,)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(label.cpu().tolist())

    # --- Compute Metrics ---------------------------
    avg_loss = total_loss / len(loader.dataset)
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    return avg_loss, accuracy, macro_f1
