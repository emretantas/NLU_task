"""
NLI Dataset for PyTorch DataLoader.

Produces batches with all tensors needed by InputEmbeddingModule and model.
"""

import torch
from torch.utils.data import Dataset
from .tokenization import tokenise


class NLIDataset(Dataset):
    """
    PyTorch Dataset for NLI pairs.

    Each item returns a dict with all tensors ready for InputEmbeddingModule + model.

    Keys returned by __getitem__:
      premise_ids      [MAX_LEN]           int64   — GloVe token indices
      hypothesis_ids   [MAX_LEN]           int64   — GloVe token indices
      premise_char     [MAX_LEN, 20]       int64   — CharCNN input
      hypothesis_char  [MAX_LEN, 20]       int64   — CharCNN input
      premise_pos      [MAX_LEN]           int64   — POS tag indices
      hypothesis_pos   [MAX_LEN]           int64   — POS tag indices
      premise_neg      [MAX_LEN, 2]        float32 — negation flags
      hypothesis_neg   [MAX_LEN, 2]        float32 — negation flags
      premise_elmo     [MAX_LEN, 1024]     float32 — pre-computed ELMo
      hypothesis_elmo  [MAX_LEN, 1024]     float32 — pre-computed ELMo
      premise_mask     [MAX_LEN]           int64   — 1=real, 0=pad
      hypothesis_mask  [MAX_LEN]           int64   — 1=real, 0=pad
      label            []                  int64   — 0 or 1

    Args:
        df: pandas DataFrame with columns: premise, hypothesis, label
        vocab: Vocabulary instance
        char2idx: Character to index mapping
        pos2idx: POS tag to index mapping
        elmo_prem: numpy array of pre-computed ELMo for premises [N, MAX_LEN, 1024]
        elmo_hyp: numpy array of pre-computed ELMo for hypotheses [N, MAX_LEN, 1024]
        max_len: Maximum sequence length (default: 64)
    """

    def __init__(self, df, vocab, char2idx, pos2idx, elmo_prem, elmo_hyp, max_len=64):
        # Import helper functions here to avoid circular dependencies
        # These are imported at runtime when dataset is created

        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.char2idx = char2idx
        self.pos2idx = pos2idx
        self.elmo_prem = elmo_prem
        self.elmo_hyp = elmo_hyp
        self.max_len = max_len

        # Pre-tokenize all text (computed once at initialization)
        self.prem_tokens = [tokenise(s) for s in df['premise']]
        self.hyp_tokens = [tokenise(s) for s in df['hypothesis']]

    def __len__(self):
        """Return number of examples in dataset."""
        return len(self.df)

    def _get_mask(self, tokens):
        """Generate mask: 1 for real tokens, 0 for padding."""
        real_len = min(len(tokens), self.max_len)
        return [1] * real_len + [0] * (self.max_len - real_len)

    def _get_token_ids(self, tokens):
        """Convert tokens to vocabulary indices with padding."""
        ids = self.vocab.encode(tokens)[:self.max_len]
        # Pad with 0 (PAD token)
        return ids + [0] * (self.max_len - len(ids))

    def __getitem__(self, idx):
        """
        Get a single example.

        Args:
            idx: Index of example to retrieve

        Returns:
            dict with all tensor keys needed by InputEmbeddingModule
        """
        # Import these functions here to avoid issues with notebook imports
        # When running from notebook, these are defined inline
        # When running as module, these should be imported from respective modules
        try:
            from elmo.char_cnn import words_to_char_tensor
            from elmo.pos_embedding import get_pos_ids
            from elmo.negation import get_negation_flags
        except ImportError:
            # If imports fail, these must be defined in the notebook already
            # This allows the notebook to work standalone
            pass

        prem_tok = self.prem_tokens[idx]
        hyp_tok = self.hyp_tokens[idx]

        return {
            # Token IDs for GloVe
            'premise_ids': torch.tensor(self._get_token_ids(prem_tok), dtype=torch.long),
            'hypothesis_ids': torch.tensor(self._get_token_ids(hyp_tok), dtype=torch.long),

            # Character IDs for CharCNN
            'premise_char': words_to_char_tensor(prem_tok, self.char2idx, 20, self.max_len),
            'hypothesis_char': words_to_char_tensor(hyp_tok, self.char2idx, 20, self.max_len),

            # POS IDs
            'premise_pos': torch.tensor(get_pos_ids(prem_tok, self.pos2idx, self.max_len), dtype=torch.long),
            'hypothesis_pos': torch.tensor(get_pos_ids(hyp_tok, self.pos2idx, self.max_len), dtype=torch.long),

            # Negation flags
            'premise_neg': get_negation_flags(prem_tok, self.max_len),
            'hypothesis_neg': get_negation_flags(hyp_tok, self.max_len),

            # Pre-computed ELMo embeddings (just slice the array)
            'premise_elmo': torch.tensor(self.elmo_prem[idx], dtype=torch.float32),
            'hypothesis_elmo': torch.tensor(self.elmo_hyp[idx], dtype=torch.float32),

            # Masks for actual (non-padded) lengths
            'premise_mask': torch.tensor(self._get_mask(prem_tok), dtype=torch.long),
            'hypothesis_mask': torch.tensor(self._get_mask(hyp_tok), dtype=torch.long),

            # Label
            'label': torch.tensor(int(self.df.loc[idx, 'label']), dtype=torch.long),
        }
