# Implementing Skip-Gram Model in Python

## Introduction

In Natural Language Processing (NLP), the skip-gram model plays a critical role
in predicting context words based on a given target word. This document provides
a detailed outline for implementing a skip-gram model, with a focus on its key
components: the Softmax function, Noise Contrastive Estimation (NCE) loss, and
Negative Sampling.

## Components of the Skip-Gram Model

1. **Softmax Function**: The softmax function is essential in converting the
   model's raw output scores, known as logits, into probabilities. This
   conversion is crucial in the output layer for multi-class classification
   tasks common in language modeling and other NLP applications. By applying the
   softmax function, the model can probabilistically evaluate each class (or
   word in the context of NLP) and make more informed predictions.

2. **Noise Contrastive Estimation (NCE) Loss**: NCE loss is particularly
   effective in training the skip-gram model, especially with large
   vocabularies. Traditional approaches to handling such vocabularies can become
   computationally prohibitive. NCE addresses this by transforming the problem
   into a binary classification: distinguishing a target word from noise words
   sampled from a noise distribution, typically the unigram distribution of
   words. This method significantly reduces the computational burden while
   maintaining effectiveness in learning word representations.

3. **Negative Sampling**: Negative Sampling, often used in tandem with the
   word2vec model, is a simplified version of NCE. It aims to reduce the
   computational load of training by focusing on a small subset of negative
   examples. Rather than predicting the probability of each word in the
   vocabulary given a context (as a full softmax would), negative sampling
   trains the model to differentiate the target word from several randomly
   chosen negative words.

## Python Implementation using NumPy

```python
import numpy as np

# Softmax Function
def softmax(logits):
    exps = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
    return exps / np.sum(exps, axis=0)

# NCE Loss Function
def nce_loss(true_logits, noise_logits):
    true_loss = -np.log(sigmoid(true_logits))
    noise_loss = -np.sum(np.log(sigmoid(-noise_logits)))
    return true_loss + noise_loss

# Sigmoid Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Example usage:
logits = np.array([...])  # Output from the model for a specific context
probabilities = softmax(logits)
true_logits = np.array([...])
noise_logits = np.array([...])
loss = nce_loss(true_logits, noise_logits)
```

## Model Operation

In the skip-gram model's workflow, the softmax function is applied during the
forward pass to convert logits into probabilities. When calculating loss, either
NCE loss or negative sampling is employed, based on the specific requirements of
the task. The computed gradients from this loss are vital for the
backpropagation process, enabling the model to learn and adjust its weights
effectively.
