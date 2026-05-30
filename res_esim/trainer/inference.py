import torch

# Label Mapping; TODO: Move to data module.
LABEL2IDX = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
IDX2LABEL = {v: k for k, v in LABEL2IDX.items()}

def predict(oracle, loader, device):
    """
    Run Inference on loader data.
    Returns a list of predicted output.
    """

    oracle.eval()

    all_preds = []

    with torch.no_grad():
        for batch in loader:
            # --- Unpack & Move to Device ------------------
            premise_embedding = batch['premise_embedding'].to(device)   # (batch, len_p, input_dim)
            hyp_embedding     = batch['hyp_embedding'].to(device)       # (batch, len
            premise_length    = batch['premise_length'].to(device)      # (batch,)
            hyp_length        = batch['hyp_length'].to(device)          # (batch,)

            # --- Forward Pass -----------------------------
            logits = oracle(
                premise_embedding, hyp_embedding, premise_length, hyp_length
            )

            preds = logits.argmax(dim=-1)
            all_preds.extend([IDX2LABEL[p.item()] for p in preds])

        return all_preds
