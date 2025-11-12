"""
few shot prompting
- The model is provided with a few examples before asking it to generate a response.
"""

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = OpenAI(
    api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

system_prompt = """
You are an AI Assistant that specialized in solving maths problem.
The user will give you a math problem. You have to analyze the problem and break down the problem step by step and finally give output.

Example:
Input: What is the result of 2 x 6
Output: 2 x 6 is 12 which is calculated by multiplying 2 by 6.

Input: sum of 12 and 3
Output: 12 + 3 is 15 which is calculated by adding 12 by 3.

Input: why is sky blue
Output: The given problem is not a math problem. You can only ask math problems.
"""

result = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {"role": "system", "content": system_prompt},
        # {"role": "user", "content": "What is result of 7x6"},
        {"role": "user", "content": "What is sky blue"},
    ],
)

print(result.choices[0].message.content)
