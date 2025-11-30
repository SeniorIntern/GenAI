"""
# Steps For Building Basic RAG:
1. Index - create chunks of file (because of context window) -> create embeddings -> store in vector database (pinecone or QDrant).
2. Retrieval - create embeddings of user's query and do similarity search in the database. Get the relevant context.
3. Generation - give user's prompt and relevant chunk to an LLM, show output to user.




## Challenges:
There are always broken chunks when chunking. For an expected chunk - "This is a full sentence", you might get "This is a" where the chunk is incomplete.

More chunk leads to hallucination, less chunk means cutting on the data.

There is also huge challenge in cross-referencing. Example - for a law paper there are cross referencing between articles i.e. article 1 depends on article 10, article 10 depends on article 24, and so on.

## Dependencies
```shell
uv add langchain_google_genai langchain_qdrant langchain_community langchain_text_splitters pypdf
uv add langchain_openai # For embedding via OpenAI model:
uv add langchain_google_genai # For embedding using Gemini model
uv add sentence_transformers # For embedding using huggin face local embedding (free, local, and no API quota)
```

## STEPS:

1. DB Setup
```shell
# open the terminal in the same directory where yml file is located and run the shell command
docker compose -f docker-compose.yml up

# Qdrant Dashboard - http://localhost:6333/dashboard
```

2. Indexing
```shell
uv run indexing.py
```
You can see the vectorized data in qdrant collection (as specified in configuration). The collection will be auto created.

3. Chat (Retrieval and Generation)
Covered in chat.py
"""

# INDEXING
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if api_key is None:
    print(f"Error getting api key {api_key}")

pdf_path = Path(__file__).parent / "nodejs.pdf"

# Loading
loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()  # Read PDF file page-by-page

# Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=400,
)

split_docs = text_splitter.split_documents(documents=docs)

# Vector Embeddings
# use OpenAI for state-of-the-art semantic accuracy for massive enterprise-scale RAG
"""
from langchain_openai import OpenAIEmbeddings

OpenAI Embeddings
==================
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=api_key
)
"""


"""
from langchain_google_genai import GoogleGenerativeAIEmbeddings

Gemini Embeddings
==================
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", google_api_key=api_key
)
"""

# local embedding (free, no API quota, offline, super fast )
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Using [embedding_model] create embeddings of [split_docs] and store in QDrant DB
vector_store = QdrantVectorStore.from_documents(
    documents=split_docs,
    url="http://localhost:6333",
    collection_name="learning_vectors",
    embedding=embedding_model,
)

print("Indexing of Documents Done...")
