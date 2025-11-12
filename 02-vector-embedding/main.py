"""
VECTOR EMBEDDING - A vector embedding is a way of representing complex data in numerical form, while preserving the semantic (structural meaning) of that data
Vector embeddings are generated from the tokens.
"""

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = OpenAI(
    api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

message = "Hello world"

# vector embedding
response = client.embeddings.create(model="text-embedding-004", input=message)
print(f"vector embedding- {response.data[0].embedding}")
