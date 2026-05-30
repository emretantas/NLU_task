"""
precompute.py
-------------
Self-contained embedding pre-computation for inference.
Wraps all logic from NLU_Embeddings_v2 notebook into a single class.

Usage:
    # On Colab A100 — upload test.csv, then run:
    from precompute import EmbeddingPrecomputer

    pc = EmbeddingPrecomputer(
        meta_path    = 'output/meta.pt',
        elmo_options = 'elmo_model/options.json',
        elmo_weights = 'elmo_model/weights.hdf5',
    )
    pc.run(
        csv_path   = 'test.csv',
        output_npz = 'inference_embeddings.npz',
    )
"""

import re
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm


# ── Constants ─────────────────────────────────────────────────────────────────
MAX_LEN      = 64
MAX_WORD_LEN = 20
GLOVE_DIM    = 300
ELMO_DIM     = 1024
CHAR_OUT_DIM = 100
POS_DIM      = 50
NEG_DIM      = 3
TOTAL_DIM    = GLOVE_DIM + ELMO_DIM + CHAR_OUT_DIM + POS_DIM + NEG_DIM  # 1477

TRIGGERS   = {'not','no','never',"n't",'neither','nor','without','lack',
              'lacking','fail','fails','failed','hardly','barely','scarcely','cannot'}
BOUNDARIES = {',','.','but','and','or','though','although'}


# ── Internal modules (same architecture as training) ─────────────────────────
class _CharCNN(nn.Module):
    """Kim et al. (2016) — 100d character-level CNN."""
    def __init__(self, char_vocab_size, char_embed_dim=30, out_dim=100):
        super().__init__()
        self.char_embedding = nn.Embedding(char_vocab_size, char_embed_dim, padding_idx=0)
        nf = out_dim // 3
        self.conv2   = nn.Conv1d(char_embed_dim, nf,     kernel_size=2)
        self.conv3   = nn.Conv1d(char_embed_dim, nf,     kernel_size=3)
        self.conv4   = nn.Conv1d(char_embed_dim, nf + 1, kernel_size=4)
        self.dropout = nn.Dropout(0.3)

    def forward(self, char_ids):
        B, S, W = char_ids.shape
        x = self.char_embedding(char_ids.view(B * S, W)).transpose(1, 2)
        def pool(conv):
            return F.max_pool1d(F.relu(conv(x)), conv(x).shape[-1]).squeeze(-1)
        out = self.dropout(torch.cat([pool(self.conv2), pool(self.conv3), pool(self.conv4)], dim=-1))
        return out.view(B, S, -1)  # [B, S, 100]


class _POSEmbedding(nn.Module):
    """Learned 50d POS tag embeddings."""
    def __init__(self, pos_vocab_size, embed_dim=50):
        super().__init__()
        self.embedding = nn.Embedding(pos_vocab_size, embed_dim, padding_idx=0)

    def forward(self, pos_ids):
        return self.embedding(pos_ids)  # [B, S, 50]


# ── Main class ────────────────────────────────────────────────────────────────
class EmbeddingPrecomputer:
    """
    Pre-computes full 1477d embeddings for any NLI CSV.

    Components:
      GloVe   [300d] — Pennington et al. (2014), frozen
      ELMo   [1024d] — Peters et al. (2018), via AllenNLP venv
      CharCNN [100d] — Kim et al. (2016)
      POS      [50d] — Universal POS tags via spaCy
      Negation  [3d] — combined rule-based + spaCy dep parse

    Args:
        meta_path    : path to meta.pt (vocab, char2idx, pos2idx, glove_matrix)
        elmo_options : path to ELMo options.json
        elmo_weights : path to ELMo weights.hdf5
        elmo_venv    : path to Python 3.10 venv with AllenNLP installed
        batch_size   : batch size for pre-computation
    """

    def __init__(
        self,
        meta_path    : str = 'output/meta.pt',
        elmo_options : str = 'elmo_model/options.json',
        elmo_weights : str = 'elmo_model/weights.hdf5',
        elmo_venv    : str = '/content/myenv/bin/python',
        batch_size   : int = 256,
    ):
        self.meta_path    = Path(meta_path)
        self.elmo_options = Path(elmo_options)
        self.elmo_weights = Path(elmo_weights)
        self.elmo_venv    = elmo_venv
        self.batch_size   = batch_size
        self._loaded      = False

    # ── Step 1: Load meta ─────────────────────────────────────────────────────
    def _load_meta(self):
        print(f'[1/5] Loading meta from {self.meta_path}...')
        meta             = torch.load(self.meta_path, map_location='cpu', weights_only=False)
        vocab_raw = meta['vocab']
        if isinstance(vocab_raw, list):
            class _Vocab:
                def __init__(self, words):
                    self.word2idx = {w: i for i, w in enumerate(words)}
            self.vocab = _Vocab(vocab_raw)
        else:
            self.vocab = vocab_raw
        self.char2idx    = meta['char2idx']
        self.pos2idx     = meta['pos2idx']
        self.glove_matrix = meta['glove_matrix']
        print(f'      vocab={len(self.vocab.word2idx)}  char={len(self.char2idx)}  pos={len(self.pos2idx)}')

    # ── Step 2: Build embedding layers ───────────────────────────────────────
    def _build_layers(self):
        print('[2/5] Building embedding layers...')

        import spacy
        self.nlp = spacy.load('en_core_web_sm')

        self.glove_layer = nn.Embedding(len(self.glove_matrix), GLOVE_DIM, padding_idx=0)
        self.glove_layer.weight = nn.Parameter(
            torch.tensor(self.glove_matrix, dtype=torch.float32), requires_grad=False)
        self.glove_layer.eval()

        self.char_cnn = _CharCNN(char_vocab_size=len(self.char2idx))
        self.char_cnn.eval()

        self.pos_embedding = _POSEmbedding(pos_vocab_size=len(self.pos2idx))
        self.pos_embedding.eval()

        print(f'      GloVe frozen  |  CharCNN  |  POS  |  spaCy en_core_web_sm')

    # ── Step 3: ELMo pre-computation (via venv) ───────────────────────────────
    def _run_elmo(self, csv_path, prem_out, hyp_out):
        print('[3/5] Pre-computing ELMo (Peters et al., 2018)...')

        script = f'''
import numpy as np, pandas as pd, re, torch
from allennlp.modules.elmo import Elmo, batch_to_ids

elmo = Elmo("{self.elmo_options}", "{self.elmo_weights}",
            num_output_representations=1, dropout=0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
elmo = elmo.to(device).eval()
print(f"ELMo on {{device}}")

def tokenise(text):
    return re.findall(r"[a-z]+(?:[\\x27][a-z]+)*", text.lower().strip())

def compute_elmo(sentences, name):
    print(f"  {{len(sentences)}} {{name}}...")
    token_lists = [tokenise(s) for s in sentences]
    token_lists = [t[:64] if len(t) > 0 else ["unk"] for t in token_lists]
    out = np.zeros((len(token_lists), 64, 1024), dtype=np.float32)
    for i in range(0, len(token_lists), 64):
        batch = token_lists[i:i+64]
        with torch.no_grad():
            emb = elmo(batch_to_ids(batch).to(device))["elmo_representations"][0].cpu().numpy()
        for j in range(len(batch)):
            sl = min(len(batch[j]), 64)
            out[i+j, :sl, :] = emb[j, :sl, :]
        if (i // 64) % 10 == 0:
            print(f"    {{min(i+64, len(token_lists))}}/{{len(token_lists)}}")
    return out

df = pd.read_csv("{csv_path}")
np.save("{prem_out}", compute_elmo(df["premise"].tolist(),   "premises"))
np.save("{hyp_out}",  compute_elmo(df["hypothesis"].tolist(), "hypotheses"))
print("ELMo done.")
'''
        script_path = '/tmp/elmo_inf_run.py'
        with open(script_path, 'w') as f:
            f.write(script)
        ret = os.system(f'{self.elmo_venv} {script_path}')
        if ret != 0:
            raise RuntimeError('ELMo pre-computation failed. Check venv has AllenNLP installed.')
        print('      ELMo saved.')

    # ── Tokenisation helpers ──────────────────────────────────────────────────
    def _tokenise(self, text):
        return re.findall(r"[a-z]+(?:'[a-z]+)*", text.lower().strip())

    def _get_mask(self, tokens):
        real = min(len(tokens), MAX_LEN)
        return [1]*real + [0]*(MAX_LEN-real)

    def _get_token_ids(self, tokens):
        unk = self.vocab.word2idx.get('<UNK>', 1)
        ids = [self.vocab.word2idx.get(t, unk) for t in tokens][:MAX_LEN]
        return ids + [0]*(MAX_LEN-len(ids))

    def _words_to_char_tensor(self, token_list):
        pad = self.char2idx.get('<PAD>', 0)
        unk = self.char2idx.get('<UNK>', 1)
        result = []
        for i in range(MAX_LEN):
            if i < len(token_list):
                word = token_list[i][:MAX_WORD_LEN]
                ids  = [self.char2idx.get(c, unk) for c in word]
                ids  = ids + [pad]*(MAX_WORD_LEN-len(ids))
            else:
                ids = [pad]*MAX_WORD_LEN
            result.append(ids)
        return torch.tensor(result, dtype=torch.long)

    def _get_pos_ids(self, token_list):
        unk = self.pos2idx.get('<UNK>', 1)
        pad = self.pos2idx.get('<PAD>', 0)
        ids = [self.pos2idx.get(t.pos_, unk) for t in self.nlp(' '.join(token_list))]
        return (ids[:MAX_LEN]+[pad]*MAX_LEN)[:MAX_LEN]

    def _get_negation_flags(self, token_list):
        tokens = [t.lower() for t in token_list[:MAX_LEN]]
        boundary_flags, in_scope = [], False
        for tok in tokens:
            if tok in TRIGGERS:   in_scope = True
            if tok in BOUNDARIES: in_scope = False
            boundary_flags.append(1.0 if in_scope else 0.0)
        doc = nlp_doc = self.nlp(' '.join(tokens))
        negated_heads = {t.head.i for t in nlp_doc if t.dep_ == 'neg'}
        dep_scope = [1.0 if t.i in negated_heads else 0.0 for t in nlp_doc]
        is_neg    = [1.0 if t in TRIGGERS else 0.0 for t in tokens]
        flags = []
        for i in range(MAX_LEN):
            if i < len(tokens):
                flags.append([
                    is_neg[i],
                    dep_scope[i] if i < len(dep_scope) else 0.0,
                    boundary_flags[i]
                ])
            else:
                flags.append([0.0, 0.0, 0.0])
        return torch.tensor(flags, dtype=torch.float32)

    # ── Step 4: Full pre-computation ─────────────────────────────────────────
    def _precompute_embeddings(self, token_lists_prem, token_lists_hyp,
                                elmo_prem, elmo_hyp, labels):
        print('[4/5] Pre-computing 1477d embeddings...')
        N = len(labels)
        prem_embs  = np.zeros((N, MAX_LEN, TOTAL_DIM), dtype=np.float32)
        hyp_embs   = np.zeros((N, MAX_LEN, TOTAL_DIM), dtype=np.float32)
        prem_masks = np.zeros((N, MAX_LEN), dtype=np.int64)
        hyp_masks  = np.zeros((N, MAX_LEN), dtype=np.int64)

        with torch.no_grad():
            for start in tqdm(range(0, N, self.batch_size), desc='  batches'):
                end = min(start + self.batch_size, N)
                idx = list(range(start, end))

                # GloVe
                prem_ids = torch.tensor([self._get_token_ids(token_lists_prem[i]) for i in idx], dtype=torch.long)
                hyp_ids  = torch.tensor([self._get_token_ids(token_lists_hyp[i])  for i in idx], dtype=torch.long)
                pg = self.glove_layer(prem_ids)
                hg = self.glove_layer(hyp_ids)

                # ELMo
                pe = torch.tensor(elmo_prem[start:end], dtype=torch.float32)
                he = torch.tensor(elmo_hyp[start:end],  dtype=torch.float32)

                # CharCNN
                pc = self.char_cnn(torch.stack([self._words_to_char_tensor(token_lists_prem[i]) for i in idx]))
                hc = self.char_cnn(torch.stack([self._words_to_char_tensor(token_lists_hyp[i])  for i in idx]))

                # POS
                pp = self.pos_embedding(torch.tensor([self._get_pos_ids(token_lists_prem[i]) for i in idx], dtype=torch.long))
                hp = self.pos_embedding(torch.tensor([self._get_pos_ids(token_lists_hyp[i])  for i in idx], dtype=torch.long))

                # Negation
                pn = torch.stack([self._get_negation_flags(token_lists_prem[i]) for i in idx])
                hn = torch.stack([self._get_negation_flags(token_lists_hyp[i])  for i in idx])

                # Concatenate → [B, 64, 1477]
                prem_embs[start:end] = torch.cat([pg, pe, pc, pp, pn], dim=-1).numpy()
                hyp_embs[start:end]  = torch.cat([hg, he, hc, hp, hn], dim=-1).numpy()
                prem_masks[start:end] = np.array([self._get_mask(token_lists_prem[i]) for i in idx])
                hyp_masks[start:end]  = np.array([self._get_mask(token_lists_hyp[i])  for i in idx])

        return prem_embs, hyp_embs, prem_masks, hyp_masks

    # ── Step 5: Save ─────────────────────────────────────────────────────────
    def _save(self, output_npz, prem_embs, hyp_embs,
              prem_masks, hyp_masks, labels):
        print(f'[5/5] Saving {output_npz}...')
        np.savez_compressed(
            output_npz,
            premise_emb     = prem_embs,
            hypothesis_emb  = hyp_embs,
            premise_mask    = prem_masks,
            hypothesis_mask = hyp_masks,
            labels          = np.array(labels, dtype=np.int64),
        )
        size = os.path.getsize(output_npz)
        print(f'      Saved: {size/1e9:.2f} GB  |  shape {prem_embs.shape}')
        nan_p = np.isnan(prem_embs).sum()
        nan_h = np.isnan(hyp_embs).sum()
        print(f'      NaN check: premise={nan_p} {"✓" if nan_p==0 else "✗"}  hypothesis={nan_h} {"✓" if nan_h==0 else "✗"}')

    # ── Public API ────────────────────────────────────────────────────────────
    def run(self, csv_path: str, output_npz: str):
        """
        Full pre-computation pipeline.

        Args:
            csv_path   : path to input CSV (must have 'premise', 'hypothesis' columns)
            output_npz : path to save compressed embeddings
        """
        csv_path   = str(csv_path)
        output_npz = str(output_npz)

        print(f'\n{"="*55}')
        print(f'  EmbeddingPrecomputer — COMP34812 NLI')
        print(f'  Input:  {csv_path}')
        print(f'  Output: {output_npz}')
        print(f'  Dim:    {TOTAL_DIM}  (GloVe+ELMo+CharCNN+POS+Neg)')
        print(f'{"="*55}\n')

        # load meta and build layers
        self._load_meta()
        self._build_layers()

        # ELMo via venv
        elmo_prem_path = '/tmp/elmo_inf_prem.npy'
        elmo_hyp_path  = '/tmp/elmo_inf_hyp.npy'
        self._run_elmo(csv_path, elmo_prem_path, elmo_hyp_path)

        # load CSV and ELMo
        df = pd.read_csv(csv_path)
        if 'label' not in df.columns:
            df['label'] = 0
        labels = df['label'].tolist()

        prem_tokens = [self._tokenise(s) for s in df['premise']]
        hyp_tokens  = [self._tokenise(s) for s in df['hypothesis']]

        elmo_prem = np.load(elmo_prem_path)
        elmo_hyp  = np.load(elmo_hyp_path)

        # compute all embeddings
        prem_embs, hyp_embs, prem_masks, hyp_masks = self._precompute_embeddings(
            prem_tokens, hyp_tokens, elmo_prem, elmo_hyp, labels)

        # save
        self._save(output_npz, prem_embs, hyp_embs, prem_masks, hyp_masks, labels)

        print(f'\n✓ Pre-computation complete — {output_npz}')
        return output_npz


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Pre-compute NLI embeddings')
    parser.add_argument('--csv',      required=True, help='Input CSV path')
    parser.add_argument('--output',   required=True, help='Output .npz path')
    parser.add_argument('--meta',     default='output/meta.pt')
    parser.add_argument('--options',  default='elmo_model/options.json')
    parser.add_argument('--weights',  default='elmo_model/weights.hdf5')
    parser.add_argument('--venv',     default='/content/myenv/bin/python')
    args = parser.parse_args()

    pc = EmbeddingPrecomputer(
        meta_path    = args.meta,
        elmo_options = args.options,
        elmo_weights = args.weights,
        elmo_venv    = args.venv,
    )
    pc.run(csv_path=args.csv, output_npz=args.output)