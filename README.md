# Vektor

Vektor is an exploratory natural language processing (NLP) toolkit focused on
delving into the mechanics of text processing and understanding through the lens
of machine learning, particularly the Transformer architecture.

## Overview

This project is both an educational journey and a testbed for experimental ideas
in NLP. It encapsulates a range of functionalities from tokenization to
encoding, with a keen interest in semantic analysis through encoded token
vectors.

## Features

- `tokenizer.py`: A module for converting text into tokens, considering subwords
  and various character sets including non-ASCII characters.
- `sparse_encoder.py`: Implements sparse encoding of tokens, a stepping stone to
  understanding text representation.
- `html_to_text.py`: Converts HTML content into Markdown format, useful for
  preprocessing steps in NLP workflows.
- `model.py`: (In progress) Aims to implement a SkipGram model for generating
  word embeddings.
- Cosine Similarity Exploration: Documents and plans for applying cosine
  similarity to token vectors, potentially revealing new insights into semantic
  relationships.

## Testing

A suite of tests ensures the reliability of functionalities provided by
`vektor`. These tests can be run with the following command:

```sh
pytest tests/
```

## Documentation

Each module and idea within `vektor` is accompanied by markdown documents,
providing insights and rationale behind the implementations.

## Setup

To install `vektor`, clone the repository and install the dependencies listed in `requirements.txt` using pip:

```sh
git clone https://github.com/teleprint-me/vektor
cd vektor
pip install -r requirements.txt
```

## Contribution

While `vektor` is a personal project, contributions or ideas that can foster the
learning experience are welcome. Feel free to fork the project or submit issues
and ideas.

## License

Vektor is MIT licensed, as found in the LICENSE file.

---

`vektor` is an ever-evolving project, reflective of the learning curve
associated with the complex field of NLP and machine learning.
