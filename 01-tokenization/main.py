"""
Tokenization is the process of breaking raw text into smaller units called tokens, which the model can process.
Example:
Input- "This is a white tiger"
Output- ["This", " is", " a", " white", " tiger"] 



TOKEN -
A token is in most cases a word, parts of word, or even character depending on the tokenizer used.
The model reads it as a single unit of processing.

CONTEXT WINDOW - maximum amount of information—measured in tokens that a LLM can process, "see", and remember at one time during a single interaction.

VOCAB SIZE - Vocabulary size refers to the total number of unique tokens that the model recognizes.

LLMs do not directly read text. They convert tokens into numbers to analyze patterns and semantic relationships.

PROCESSING PIPELINE:
===================

INPUT: This is a white tiger
    |
    TOKENIZATION - break down into small pieces (words or parts of words)
    |
    v
TOKENS: This, is, a, white, tiger
    |
    v
TOKEN IDs: [1231, 13213, 1321231]

Next Step: Vector Embeddings for the token IDs.
Embeddings are high-dimensional vectors representing tokens numerically.

Why token ids are converted into vector embeddings?
Tokens are converted into vectors because neural networks can only operate on numbers, and meaning is represented as patterns in high-dimensional space.

When the tokenizer outputs: "tiger" -> 12345
That number 12345 has no meaning.

Neural networks work by:
- Matrix multiplication
- Dot products
- Distance calculations
- Linear transformations

For that to work, words must:
- Have direction
- Have magnitude
- Have geometric relationships
- That’s what embeddings provide.
"""

import tiktoken

encoder = tiktoken.encoding_for_model("gpt-4o")

print("vocab size- ", encoder.n_vocab)  # vocab size-  200019

text = "Nikhil"

tokens = encoder.encode(text)
print("Token: ", tokens)  # Token:  [45, 18226, 311]

decoded = encoder.decode(tokens)
print("Decoded Text: ", decoded)  # Decoded Text:  Nikhil
