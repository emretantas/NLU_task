"""
Tokenization utilities for NLI text processing.
"""

import re


def tokenise(text):
    """
    Lowercase and tokenise into word strings.
    Handles contractions (don't, I'm) and strips punctuation.

    Args:
        text: Input string to tokenize

    Returns:
        List of lowercase token strings

    Examples:
        >>> tokenise("I don't think it's correct.")
        ['i', "don't", 'think', "it's", 'correct']
    """
    return re.findall(r"[a-z]+(?:'[a-z]+)*", text.lower().strip())
