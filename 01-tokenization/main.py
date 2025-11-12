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

print("vocab side- ", encoder.n_vocab)

text = "Nikhil"

tokens = encoder.encode(text)
print("Token: ", tokens)

decoded = encoder.decode(tokens)
print("Decoded Text: ", decoded)
