"""
vektor/encoders/bpe.py

Paper: https://paperswithcode.com/method/bpe
Blog: https://leimao.github.io/blog/Byte-Pair-Encoding/
Code: https://github.com/rsennrich/subword-nmt
"""

import collections
import re
from typing import Dict, List, Optional, Tuple


class BytePairEncoding:
    def __init__(
        self,
        n_merges: Optional[int] = 10000,
        stop_token: Optional[str] = "</w>",
        unk_token: Optional[str] = "</u>",
    ):
        """
        Initialize the BytePairEncoding class.

        Args:
            n_merges (int): The number of merges to perform during tokenization.
        """
        self.n_merges = n_merges
        self.vocab = collections.defaultdict(int)
        self.stop = f" {stop_token}"
        self.unknown = unk_token

    def train(self, corpus: List[str]) -> None:
        """
        Train the BytePairEncoding model on a given corpus.

        Args:
            corpus (list): List of strings representing the text corpus.
        """
        self.set_vocab(corpus)

        for _ in range(self.n_merges):
            pairs = self.get_stats()
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self.merge_tokens(best)

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

    def get_token_length(self, token: str) -> int:
        """
        Calculate the length of a token, accounting for the stop token if present.

        Args:
            token (str): The token whose length needs to be calculated.

        Returns:
            int: The length of the token.
        """
        if token.endswith(self.stop):
            # Adjust length to account for stop token
            return len(token) - len(self.stop) + 1
        return len(token)

    def sort_tokens(self, tokens_frequencies: Dict[str, int]) -> List[str]:
        """
        Sort tokens based on their frequencies and token lengths.

        Args:
            tokens_frequencies (Dict[str, int]): A dictionary mapping tokens to their frequencies.

        Returns:
            List[str]: A list of sorted tokens.
        """
        sorted_tokens_tuple = sorted(
            tokens_frequencies.items(),
            key=lambda item: (self.get_token_length(item[0]), item[1]),
            reverse=True,
        )
        return [token for (token, freq) in sorted_tokens_tuple]

    def get_token_info(self) -> Tuple[Dict[str, int], Dict[str, List[str]]]:
        """
        Retrieve token frequency information and tokenization mapping.

        Returns:
            tuple: A tuple containing two dictionaries - frequency_map and token_map.
                - frequency_map: A dictionary mapping tokens to their frequencies.
                - token_map: A dictionary mapping original words to their tokenized representations.
        """
        frequency_map = collections.defaultdict(int)
        token_map: Dict[str, List[str]] = {}

        for word, frequency in self.vocab.items():
            split_word = word.split()  # Use space delimiter
            original_word = "".join(split_word)  # Omit space delimiter
            for token in split_word:
                frequency_map[token] += frequency
            token_map[original_word] = split_word

        return frequency_map, token_map

    def set_vocab(self, corpus: List[str]) -> None:
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

    def get_stats(self) -> Dict[Tuple[str, str], int]:
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

    def merge_tokens(self, pair: Tuple[str, str]) -> None:
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
    bpe = BytePairEncoding(n_merges=100)
    corpus = [
        "This is a sample text.",
        "Byte-Pair Encoding is cool!",
        "There is a cool breeze today.",
    ]

    # Train the BPE model with the corpus
    bpe.train(corpus)

    # Tokenize a new sentence using the trained model
    tokenized_sentence = bpe.tokenize("Encoding is fun!")

    # Print the tokenized result
    print("Tokenized Sentence:", tokenized_sentence)

    # Optional: Print additional information for analysis
    token_frequencies, token_map = bpe.get_token_info()
    print("\nToken Frequencies:", token_frequencies)
    print("\nTokenization Map:", token_map)
