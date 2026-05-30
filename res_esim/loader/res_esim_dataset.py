from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

class ResESIM_Dataset(Dataset):
    """
    Loads pre-computed full embedding tensors for OracleNet.
    GloVe [300d] + ELMo [1024d] + CharCNN [100d] + POS [50d] + Negation [3d] = 1477d
    Args:
        npz_path : path to .npz file (train_embeddings.npz or dev_embeddings.npz)
    """
    def __init__(self, npz_path: Path):
        print(f'Loading {npz_path}...')
        d = np.load(npz_path)
        self.prem_emb     = torch.tensor(d['premise_emb'],     dtype=torch.float32)
        self.hyp_emb      = torch.tensor(d['hypothesis_emb'],  dtype=torch.float32)
        self.prem_lengths = torch.tensor(d['premise_mask'],    dtype=torch.long).sum(dim=-1)
        self.hyp_lengths  = torch.tensor(d['hypothesis_mask'], dtype=torch.long).sum(dim=-1)
        self.labels       = torch.tensor(d['labels'],          dtype=torch.long)
        print(f'  {len(self.labels)} samples, dim={self.prem_emb.shape[-1]}')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'premise_embedding': self.prem_emb[idx],
            'hyp_embedding':     self.hyp_emb[idx],
            'premise_length':    self.prem_lengths[idx],
            'hyp_length':        self.hyp_lengths[idx],
            'label':             self.labels[idx],
        }