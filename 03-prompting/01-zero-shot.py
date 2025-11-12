"""
zero shot prompting
- The model is given a direct question or task without prior examples.
"""

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = OpenAI(
    api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

result = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {"role": "user", "content": "What is result of 7x6"},
        # {"role": "user", "content": "What is sky blue"}
    ],
)

print(result.choices[0].message.content)
