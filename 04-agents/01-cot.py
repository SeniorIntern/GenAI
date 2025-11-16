"""
# Agents
LLM works on pre trained data. They have knowledge cutoff. Thus, they don't have real time data.


## What is an AI agent?
An AI agent is an autonomous decision-making system that perceives inputs (data, APIs, sensors), reasons using models or rules, and performs actions (scripts, tools, automation) to achieve defined objectives.


It is like a digital assistant that doesn’t just give information but actually completes tasks on your behalf — for example booking tickets, writing and sending emails, searching for deals, or monitoring data automatically.

Analogy - AI Agent is like giving ChatGPT a mission, and it figures out how to complete it step by step — using tools, memory, planning.

Main purpose of Agent - Solve complex tasks through planning, decision-making, and iteration.
Info - Tool calling is a core part of AI Agent behavior

## AI Agent vs Normal LLM (ChatGPT Q&A Mode)
“ChatGPT gives answers; an AI agent gets things done.”

Unlike normal AI chat that only replies, an AI agent can take steps, make decisions, and perform tasks automatically.

Normal LLM: Q&A model doesn’t interact with external systems (unless tools enabled)
AI agent: Can run code, browse internet, call APIs, automate computer.

Normal LLM: Single message → single reply
AI agent: Multi-step planning + loops

## Mini Example in Real Life
"Find the cheapest iPhone 16 in UAE, notify me if price < AED 3500, and order automatically."

An agent would:
- Scrape multiple stores daily
- Compare with price threshold
- Notify you
- Perform checkout with saved credentials
"""

from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = OpenAI(
    api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


def getDateTimeByCityName(cityName: str):
    print("🧪 getDateTimeByCityName called...")
    return f"The current data and time of {cityName.strip()} is 16th Nov 2025, and 1:40 AM respectively."


def getCurrentWeatherByCityName(cityName: str):
    print("🧪 getCurrentWeatherByCityName called...")
    return f"The current data and time of {cityName.strip()} is 21degree Celcius, Clear Sky."


system_prompt = """
    You are a helpful AI Assistant who is specialized in resolving user query.
    You work on start, plan, action, observe mode.

    For the given user query and available tools, plan the step by step execution, based on the planning,
    select the relevant tool from the available tool. And based on the tool selected you perform an action to call the tool.

    Wait for the observation and based on the observation from the tool call resolve the user query.

    Rules:
    - Follow the Output JSON Format.
    - Always perform one step at a time and wait for the next input.
    - Carefully analyse the user query.

    Output JSON Format:
    {{
        "step": "string",
        "content": "string",
        "function": "The name of function if the step is action",
        "input": "The input parameter for the function"
    }}

    Available Tools:
    - "getCurrentWeatherByCityName": Takes a city name as input and returns the weather of the city.
    - "getDateTimeByCityName" : Takes a city name as input and returns the date time of the city.

    Example:
    Input: What is the weather in Kathmandu?
    Output: {{ "step": "plan", "content": "The user is interested in weather data of Kathmandu. So I will use the getCurrentWeatherByCityName tool to get the weather data of Kathmandu." }}
    Output: {{ "step": "action", "function": "getCurrentWeatherByCityName", "input": "Kathmandu" }}
    Output: {{ "step": "observe", "content": "24 degrees C" }}
    Output: {{ "step": "output", "content": "The weather for Kathmandu seems to be 24 degrees C" }}
"""

result = client.chat.completions.create(
    model="gemini-2.5-flash",
    response_format={"type": "json_object"},
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "What is current weather for dubai"},
        # Chain of thought prompts
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "step": "plan",
                    "content": "The user is asking for the current weather in Dubai. I need to use the `getCurrentWeatherByCityName` tool to get this information.",
                }
            ),
        },
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "step": "action",
                    "function": "getCurrentWeatherByCityName",
                    "input": "Dubai",
                }
            ),
        },
        {
            "role": "assistant",
            "content": json.dumps(
                {"step": "observe", "content": "30 degrees C, Sunny"}
            ),
        },
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "step": "output",
                    "content": "The current weather in Dubai is 30 degrees C and Sunny.",
                }
            ),
        },
    ],
)

print(result.choices[0].message.content)
