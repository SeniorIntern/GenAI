"""
Retrieval
- For given user's query, The query needs to be vectorized in order to do semantic searching in vector db. Then search for relevant information in the database.


Generation
- Use that information to guide the LLM in generating a more accurate and informative response.
"""

import os

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

load_dotenv()

# api_key = os.getenv("OPENAI_API_KEY")
api_key = os.getenv("GEMINI_API_KEY")


# Source - https://stackoverflow.com/a
# Posted by Alec Segal, modified by community. See post 'Timeline' for change history
# Retrieved 2025-11-30, License - CC BY-SA 4.0
# INFO: Fix warning for disabling tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Vector Embeddings
"""
from langchain_openai import OpenAIEmbeddings

OpenAI Embeddings
==================
embedd_fn = OpenAIEmbeddings(model="text-embedding-3-large")
"""


"""
from langchain_google_genai import GoogleGenerativeAIEmbeddings

Gemini Embeddings
==================
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", google_api_key=api_key
)
"""

# free embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_vectors",
    embedding=embedding_model,
)

# Take User Query
query = input("> ")

# Vector Similarity Search [query] in DB
relevant_chunks = vector_db.similarity_search(query=query)

# get chunks for result in text
# print("Relevent Chunks", relevent_chunks)

# client = OpenAI(api_key=api_key)
client = OpenAI(
    api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

SYSTEM_PROMPT = f"""
You are an helpful AI Assistant who responds based on the available context.

Context: Based on this given data. answer the user's query. If the answer is not available, then say "Sorry, I can only answer for questions related to NodeJS."

{relevant_chunks}

Example: 
Input: How to run nodejs code?
Output: You can run a Node.js script using the node command. Open up a new terminal window
and navigate to the directory where the script lives. From the terminal, you can use the
node command to provide the path to the script that should run.

Example: 
Input: What is the behaviour of this in arrow functions?
Output: Arrow functions don not bind their own this value. Instead, the this value of the scope in
which it was defined is accessible. This makes arrow functions bad candidates for
methods, as this won not be a reference to the object the method is defined on.

Example:
What is the longest river of the world?
Output: Sorry, I can only answer for questions related to NodeJS.
"""

result = client.chat.completions.create(
    # model="gpt-4o",
    model="gemini-2.5-flash",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ],
)

print(f"🤖: {result.choices[0].message.content}")
# print(f"🤖: {parsed_response.get('output')}")
