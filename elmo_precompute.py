
import numpy as np
import pandas as pd
import re
import torch
from pathlib import Path
from tqdm import tqdm
from allennlp.modules.elmo import Elmo, batch_to_ids

# Use paths from configuration
TRAIN_PATH   = Path("/Users/vpremakantha/Documents/UOM/Y3/NLU/NLU-CW/data/train.csv")
DEV_PATH     = Path("/Users/vpremakantha/Documents/UOM/Y3/NLU/NLU-CW/data/dev.csv")
ELMO_OPTIONS = Path("/Users/vpremakantha/Documents/UOM/Y3/NLU/NLU-CW/bin/elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json")
ELMO_WEIGHTS = Path("/Users/vpremakantha/Documents/UOM/Y3/NLU/NLU-CW/bin/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5")
OUTPUT_DIR   = Path("/Users/vpremakantha/Documents/UOM/Y3/NLU/NLU-CW/output")
MAX_LEN      = 64
BATCH_SIZE   = 64

OUTPUT_DIR.mkdir(exist_ok=True)

print("Loading ELMo (Peters et al., 2018) via AllenNLP...")
elmo   = Elmo(str(ELMO_OPTIONS), str(ELMO_WEIGHTS), num_output_representations=1, dropout=0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print(f"===== Device: {device} =====")

elmo   = elmo.to(device).eval()
print(f"ELMo loaded on {device}")

def tokenise(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)*", text.lower().strip())

def load_csv(path):
    df = pd.read_csv(path)
    return df["premise"].tolist(), df["hypothesis"].tolist()

def compute_elmo(sentences, name):
    token_lists = [tokenise(s) for s in sentences]
    token_lists = [t[:MAX_LEN] if len(t) > 0 else ["unk"] for t in token_lists]
    all_embeddings = np.zeros((len(token_lists), MAX_LEN, 1024), dtype=np.float32)
    batches = range(0, len(token_lists), BATCH_SIZE)
    for i in tqdm(batches, desc=name, unit="batch"):
        batch    = token_lists[i:i+BATCH_SIZE]
        char_ids = batch_to_ids(batch).to(device)
        with torch.no_grad():
            output = elmo(char_ids)
        emb = output["elmo_representations"][0].cpu().numpy()
        for j in range(len(batch)):
            seq_len = min(len(batch[j]), MAX_LEN)
            all_embeddings[i+j, :seq_len, :] = emb[j, :seq_len, :]
    return all_embeddings

train_prem, train_hyp = load_csv(TRAIN_PATH)
np.save(OUTPUT_DIR / "elmo_train_prem.npy", compute_elmo(train_prem, "train premises"))
np.save(OUTPUT_DIR / "elmo_train_hyp.npy",  compute_elmo(train_hyp,  "train hypotheses"))
print("Train saved.")

dev_prem, dev_hyp = load_csv(DEV_PATH)
np.save(OUTPUT_DIR / "elmo_dev_prem.npy", compute_elmo(dev_prem, "dev premises"))
np.save(OUTPUT_DIR / "elmo_dev_hyp.npy",  compute_elmo(dev_hyp,  "dev hypotheses"))
print("Dev saved.")
print("ALL DONE.")
