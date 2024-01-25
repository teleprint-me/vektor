"""
vektor/encoders/bpe.py

Paper: https://paperswithcode.com/method/bpe
Blog: https://leimao.github.io/blog/Byte-Pair-Encoding/
Code: https://github.com/rsennrich/subword-nmt
"""

import collections
import re


class BytePairEncoding:
    def __init__(self, num_merges=1000):
        """
        Initialize the BytePairEncoding class.

        Args:
            num_merges (int): The number of merges to perform during tokenization.
        """
        self.num_merges = num_merges
        self.vocab = {}  # Initialize the vocabulary

    def train(self, corpus):
        """
        Train the BytePairEncoding model on a given corpus.

        Args:
            corpus (list): List of strings representing the text corpus.
        """
        # Initialize the vocabulary with single characters as tokens
        self.vocab = {char: freq for text in corpus for char in text}

        # Perform BPE merges for a specified number of iterations
        for _ in range(self.num_merges):
            pairs = self._get_stats()
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self._merge_tokens(best)

    def tokenize(self, text):
        """
        Tokenize a text using the trained BytePairEncoding model.

        Args:
            text (str): The input text to tokenize.

        Returns:
            list: List of tokens.
        """
        # Initialize the list of tokens with individual characters
        tokens = list(text)

        # Perform tokenization using the trained vocabulary
        tokens = self._tokenize_with_vocab(tokens)

        return tokens

    def _get_stats(self):
        # Calculate token pair frequencies based on the current vocabulary
        pairs = {}
        for token, freq in self.vocab.items():
            symbols = token.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pairs[pair] = pairs.get(pair, 0) + freq
        return pairs

    def _merge_tokens(self, pair):
        # Merge the most frequent token pair in the vocabulary
        token1, token2 = pair
        merged_token = token1 + token2
        self.vocab[merged_token] = self.vocab.get(merged_token, 0)
        del self.vocab[token1]
        del self.vocab[token2]

    def _tokenize_with_vocab(self, tokens):
        # Tokenize text using the trained vocabulary
        new_tokens = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            while i < len(tokens) - 1 and (token + tokens[i + 1] in self.vocab):
                token += tokens[i + 1]
                i += 1
            new_tokens.append(token)
            i += 1
        return new_tokens
