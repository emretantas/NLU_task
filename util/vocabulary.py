"""
Vocabulary class for word-to-index mapping.
"""

from collections import Counter


class Vocabulary:
    """
    Word-to-index mapping built from training data only.

    Index 0 is reserved for <PAD>, index 1 for <UNK>.
    Words appearing fewer than min_freq times map to <UNK>.

    Args:
        min_freq: Minimum frequency threshold for including a word (default: 2)

    Example:
        >>> vocab = Vocabulary(min_freq=2)
        >>> vocab.build([['the', 'cat'], ['the', 'dog'], ['a', 'cat']])
        >>> vocab.encode(['the', 'cat', 'bird'])
        [2, 3, 1]  # 'bird' maps to UNK
    """

    def __init__(self, min_freq=2):
        PAD_TOKEN = '<PAD>'
        UNK_TOKEN = '<UNK>'

        self.min_freq = min_freq
        self.word2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.idx2word = {0: PAD_TOKEN, 1: UNK_TOKEN}
        self.word_freq = Counter()

    def build(self, token_lists):
        """
        Build vocabulary from a list of token lists (training data only).

        Args:
            token_lists: List of token lists, e.g., [['the', 'cat'], ['a', 'dog']]
        """
        # Count frequencies
        for tokens in token_lists:
            self.word_freq.update(tokens)

        # Add words meeting frequency threshold
        for word, freq in self.word_freq.items():
            if freq >= self.min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

        print(f'Vocabulary size: {len(self.word2idx)} tokens (min_freq={self.min_freq})')

    def encode(self, tokens):
        """
        Convert token list to indices.

        Args:
            tokens: List of token strings

        Returns:
            List of integer indices
        """
        unk = self.word2idx['<UNK>']
        return [self.word2idx.get(t, unk) for t in tokens]

    def __len__(self):
        """Return vocabulary size."""
        return len(self.word2idx)

    def __repr__(self):
        return f"Vocabulary(size={len(self)}, min_freq={self.min_freq})"
