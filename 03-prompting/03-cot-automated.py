"""
Chain of Thought
The model is encouraged to break down reasoning step by step before arriving at an answer. This results in a very good response.
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = OpenAI(
    api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

system_prompt = """
You are an AI Assistant that specialized in solving maths problem.
The user will give you a math problem. You have to analyze the problem and break it down into steps. 

First you will check if it is a math problem. If it is, you will first analyse the problem, think of solution, calculate the output, validate if the answer is correct, and finally show result.

Follow the steps in sequence that is: "analyse", "think", "output", "validate", "result"

Output Format: 
{{ step: "string", content: "string" }}

Rules:
1. Follow strict JSON output as per output schema
2. Always perform one step at a time and wait for next input
3. Carefully analyse the user query

Example:
Input: What is 2 x 6
Output: {{ step: "analyse", content: "The user is interested in the result of 2 x 6" }}
Output: {{ step: "think", content: "The solve the problem I have mulitply 2 by 6" }}
Output: {{ step: "output", content: "The output of 2 x 6 is 12" }}
Output: {{ step: "validate", content: "The result 12 seems to be correct" }}
Output: {{ step: "result", content: "The result of 2 x 6 is 12." }}

Input: sum of 12 and 3
Output: {{ step: "analyse", content: "The user is interested in the result of 12 + 3" }}
Output: {{ step: "think", content: "The solve the problem I have add 12 and 3" }}
Output: {{ step: "output", content: "The output of 12 + 3 is 15" }}
Output: {{ step: "validate", content: "The result 15 seems to be correct" }}
Output: {{ step: "result", content: "The result of 12 + 3 is 15." }}

Input: why is sky blue
Output: {{ step: "result", content: "The given problem is not a math problem. Sorry, I can only help you for math problems" }}
"""


def run_cot():
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    while True:
        result = client.chat.completions.create(
            model="gemini-2.5-flash",
            response_format={"type": "json_object"},
            messages=messages,  # pyright: ignore
        )
        parsed_result = json.loads(result.choices[0].message.content)  # pyright: ignore
        # print(parsed_result)
        # {'step': 'analyse', 'content': 'The user is interested in the result of 3 + 2.'}

        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(parsed_result),
            }
        )
        if parsed_result.get("step") == "result":
            print(f"🤖: {parsed_result.get('content')} ")
            break
        else:
            print(f"💭: {parsed_result.get('content')} ")


print("🤖 How can I help you...")
query = input("> ")

if query.strip() == "":
    print("You need to provide a query")
else:
    run_cot()
