"""
TOKEN - chunk of text. represented as number because that's what the model understands
A token is typically a word, subword, or even character depending on the tokenizer used.
The model reads it as a single unit of meaning.

CONTEXT WINDOW - maximum number of tokens the model can consider at once when generating or understanding text.

VOCAB SIZE - Vocabulary size refers to the total number of unique tokens that the model recognizes and can process.
Vocabulary size refers to the total number of unique tokens that the model recognizes and can process.

FLOW:
1. Raw text -> tokens -> ["The", " cat", " sat", " on", " the", " mat"]
2. Tokens -> token IDs -> [976, 9059, 10139, 402, 290, 2450].
        These numbers are IDs of the tokens in the model's internal vocabulary.
3. Token IDs -> vectors (embedding lookup) -> list of dense vectors
4. Model uses those vectors for meaning, similarity, or prediction -> final embedding or next-token probabilities
"""

import tiktoken

encoder = tiktoken.encoding_for_model("gpt-4o")

print("vocab size- ", encoder.n_vocab)  # vocab size-  200019

text = "Nikhil"

tokens = encoder.encode(text)
print("Token: ", tokens)  # Token:  [45, 18226, 311]

decoded = encoder.decode(tokens)
print("Decoded Text: ", decoded)  # Decoded Text:  Nikhil
