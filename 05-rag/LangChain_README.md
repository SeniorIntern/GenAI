# LangChain

LangChain is a library that handles the whole chain(multiple steps) when building a LLM powered app.

With LangChain we don't need to:

- Define functions for handling files.
- Generate embedding for each model:

An example RAG chain: loading a document, chunking, embedding, storing vector, etc.

With LangChain we can:

```python
loader = pdf_loader(path)
splitter = text_splitter(loader.load())
embedding = openAI()
qdrant = Qdrant()

chain = loader | splitter | embedding | qdrant
chain.invoke(pdf_path)
```
