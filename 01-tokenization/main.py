"""
TOKEN -
A token is in most cases a word, parts of word, or even character depending on the tokenizer used.
The model reads it as a single unit of meaning.

CONTEXT WINDOW - maximum amount of information—measured in tokens that a LLM can process, "see", and remember at one time during a single interaction.

VOCAB SIZE - Vocabulary size refers to the total number of unique tokens that the model recognizes.

LLMs do not directly read text. They convert tokens into numbers to analyze patterns and semantic relationships.

FLOW:
INPUT: This is a white tiger
    |
    TOKENIZATION - break down into small pieces (words or parts of words)
    |
    v
TOKENS: This, is, a, white, tiger
    |
    EMBEDDINGS - Turn Token into numerical ID/representation capturing their meaning using vocabulary
    |
    v
[1231, 13213, 1321231]
"""

import tiktoken

encoder = tiktoken.encoding_for_model("gpt-4o")

print("vocab size- ", encoder.n_vocab)  # vocab size-  200019

text = "Nikhil"

tokens = encoder.encode(text)
print("Token: ", tokens)  # Token:  [45, 18226, 311]

decoded = encoder.decode(tokens)
print("Decoded Text: ", decoded)  # Decoded Text:  Nikhil
