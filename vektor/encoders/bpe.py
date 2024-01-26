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
        self.vocab = collections.defaultdict(int)
        self.stop = " </w>"

    def train(self, corpus):
        """
        Train the BytePairEncoding model on a given corpus.

        Args:
            corpus (list): List of strings representing the text corpus.
        """
        self._set_vocab(corpus)

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
        return self._tokenize_with_vocab(tokens)

    def get_token_info(self):
        """
        Retrieve token frequency information and tokenization mapping.

        Returns:
            tuple: A tuple containing two dictionaries - frequency_map and token_map.
                - frequency_map: A dictionary mapping tokens to their frequencies.
                - token_map: A dictionary mapping original words to their tokenized representations.
        """
        frequency_map = collections.defaultdict(int)
        token_map = {}

        for word, frequency in self.vocab.items():
            split_word = word.split()  # Use space delimiter
            original_word = "".join(split_word)  # Omit space delimiter
            for token in split_word:
                frequency_map[token] += frequency
            token_map[original_word] = split_word

        return frequency_map, token_map

    def _set_vocab(self, corpus: list[str]) -> None:
        """
        Set the vocabulary based on the given corpus.

        Args:
            corpus (list): List of strings representing the text corpus.

        Raises:
            ValueError: If corpus is empty or contains non-string elements.
        """
        if not corpus or not all(isinstance(text, str) for text in corpus):
            raise ValueError("Corpus must be a list of strings.")

        # Ensure vocab is reset upon a new corpus
        self.vocab = collections.defaultdict(int)

        # Break down corpus into lines.
        for line in corpus:
            # Break down the line into words.
            for word in line.split():
                # Group space-separated characters by bounding them with a stop token.
                token = " ".join(list(word)) + self.stop
                # Add token to vocab using a unique integer.
                self.vocab[token] += 1

    def _get_stats(self):
        """
        Calculate token pair frequencies based on the current vocabulary.

        Returns:
            dict: A dictionary mapping token pairs to their frequencies.
        """
        pairs = collections.defaultdict(int)
        for token, freq in self.vocab.items():
            symbols = token.split()
            for i in range(len(symbols) - 1):
                current_symbol = symbols[i]
                next_symbol = symbols[i + 1]
                pair = (current_symbol, next_symbol)
                pairs[pair] += freq
        return pairs

    def _merge_tokens(self, pair: tuple[str, str]) -> None:
        """
        Merge tokens in the vocabulary based on a given pair.

        Args:
            pair (tuple): A tuple containing two strings representing the pair of tokens to merge.
        """
        output_vocab = collections.defaultdict(int)
        lookbehind, group, lookahead = r"(?<!\S)", re.escape(" ".join(pair)), r"(?!\S)"
        bigram_pattern = re.compile(lookbehind + group + lookahead)

        for token, freq in self.vocab.items():
            merged_token = bigram_pattern.sub("".join(pair), token)
            # Keep an eye on += freq; this might create a subtle bug.
            output_vocab[merged_token] += freq

        self.vocab = output_vocab

    def _tokenize_with_vocab(self, tokens):
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


if __name__ == "__main__":
    # Example usage:
    bpe = BytePairEncoding(num_merges=100)
    corpus = ["This is a sample text.", "Byte-Pair Encoding is cool!"]
    bpe.train(corpus)
    print(bpe.tokenize("Encoding is fun!"))
