# Retrieval-Augmented Generation (RAG)

RAG is an AI technique that combines the power of information retrieval with LLM to enhance the accuracy and context-awareness of AI-generated response.

With RAG, the system provide more accurate, up-to-date, and factual responses by grounding them in external data.

## Analogy

RAG is like Googling something and feeding the search results to ChatGPT

## Common Scenario for a Business

Chat Application - The data is stored in a PG database. This data will be used for chat response. The model is based on pre-trained data.

SOLUTIONS:

1. Fine Tuning - Take dump of the data daily and fine tune the model.
   Problem: Expensive, Time consuming, Not real-time (24 hours delay), can not write data

2. RAG (best solution for business use case) - Using function calling we can fetch real time data and write to db.

## How RAG Works

It works by first searching for relevant information from external data sources (like database, documents, or the web) and then using that information to guide the LLM in generating a more accurate and informative response.

Basically, put relevant data in prompt/context. For every data we fetch, we ingest the data in context window so LLMs get context.

Challenging part - Getting the right data from one or multiple source with huge data size (TBs of data) and ingest in context window.

## RAG In Action

Query: User asks a question (e.g., _"What is the latest revenue of Tesla?"_).

1. **Retrieval**: The system fetches relevant documents or snippets from a knowledge base, database, or search engine (e.g., internal docs, websites, PDFs).
2. **Augmented**: These retrieved texts are passed as _context_ to the language model.
3. **Generation**: The language model (like GPT) uses this context to craft an informed and relevant answer.

## Basic RAG Setup

### Ingestion

1. Chunking - Index data source so it'll be easier to get the piece containing the information. The pdf will be in chunks.
2. Embedding - Generate vector embeddings of information. Use LLM to generate semantic meaning of the chunks.
3. Store - store the embeddings in vector database. Along with vector embedding you can store the metadata. The metadata can contain information like page number.

### Inference

1. Embedding - Generate vector embedding of user's query.
2. Search - Search the query's embedding in previous vector store/db. Once you found the most semantic embedding, find the nearby vector embeddings also and get their page numbers.

Once you get the relevant chunks, instruct the LLM to read the corresponding page numbers of the chunks.
Example - Read page 1 and 2. Re-run the user's query based on the pages found. Example - Give page 1 and 2 to LLM context and answer.
