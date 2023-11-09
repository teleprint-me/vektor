"""
tests/test_skip_gram_model.py
"""

import pytest

# Assuming your skip-gram model is in a module named model
from vektor.model import SkipGramModel


@pytest.fixture
def small_corpus():
    return [
        "the quick brown fox jumps over the lazy dog",
        "I wish to wish the wish you wish to wish",
        "but if you wish the wish the witch wishes I won't wish the wish you wish to wish",
    ]


def test_skipgram_training(small_corpus):
    # Initialize the SkipGram model with suitable parameters
    window_size = 2  # The size of the context window
    vector_dim = 50  # The size of the word embedding vectors
    model = SkipGramModel(window_size=window_size, vector_dim=vector_dim)

    # Train the model on the small_corpus
    # This is a placeholder for the actual training function
    model.train(small_corpus)

    # Now we test if the model has learned something
    # We will need to have a method to get the vector for a word and to get the context words
    target_word = "wish"
    context_words = ["to", "the"]  # This is an expected context in the corpus

    # Assuming we have a method in our model to get the closest context words
    predicted_context = model.get_context_words(target_word, top_n=2)

    assert set(predicted_context) == set(
        context_words
    ), f"Expected context {context_words} but got {predicted_context}"
