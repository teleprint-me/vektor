# Vektor Project Documentation

## Overview

**Project Name:** Vektor

**Purpose:** The Vektor project is a personal endeavor aimed at understanding
and implementing Transformer models from the ground up. It focuses on the core
concepts of tokenization, encoding, and the architecture of Transformer models,
particularly for Natural Language Processing (NLP) tasks.

## Project Goals

1. **Understand Transformer Models:** Deepen understanding of Transformer
   architectures, including attention mechanisms and layer structures.
2. **Custom Tokenization and Encoding:** Develop efficient, unique, and
   potentially language-agnostic methods for tokenization and encoding.
3. **Model Experimentation:** Experiment with unconventional ideas in model
   architecture to explore and learn.
4. **Modular and Scalable Design:** Build a modular system that allows for
   extension to multi-modal applications and is scalable for different use
   cases.
5. **Focus on Text Generation:** Prioritize development for applications in text
   generation.

## Current Progress

### Codebase Structure

- **Activations**: Implementation of activation functions like ReLU, Sigmoid,
  and Tanh.
- **Base**: Core modules including base layer, loss functions, and the neural
  network framework.
- **CLI Tools**: Tools for tasks like HTML to text conversion.
- **Encoders**: Modules for tokenization and various encoding strategies.
- **Errors**: Error calculation modules like Cross-Entropy, Mean Squared Error,
  and Root Mean Squared Error.
- **Layers**: Different layer implementations including Dense, Linear, and
  Sparse layers.
- **Models**: Current models in progress, including the Skip-Gram model.

### Development Highlights

- Reuse of code from previous project `tmnn` for foundational elements.
- Implementation of basic neural network components and activation functions.
- Development of custom tokenization and encoding methods.
- Initiation of Skip-Gram model implementation.

## Future Roadmap

### Short-Term Goals

- Complete the implementation of the Skip-Gram model.
- Expand the `layers` module with additional types of neural network layers.
- Develop and integrate more advanced encoding techniques.

### Long-Term Goals

- Explore and implement attention mechanisms and other core Transformer
  components.
- Experiment with multi-modal capabilities and integrate them into the existing
  framework.
- Optimize the system for performance and scalability.
- Evaluate the model's effectiveness on various NLP tasks.

## Challenges and Learnings

- Transition from C++ to Python for ease of development and rapid prototyping.
- Balancing exploration with focus on core project goals.
- Implementing efficient data structures and algorithms for NLP tasks.

## Conclusion

Vektor is an ongoing project that encapsulates the journey of learning and
building a Transformer model from scratch. It serves as a testament to the power
of hands-on learning and the pursuit of knowledge in the field of AI and NLP.
