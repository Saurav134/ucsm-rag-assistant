from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from sentence_transformers import CrossEncoder

from llm_client import stream_llm

load_dotenv()

app = FastAPI()

# -----------------------
# Conversation memory
# -----------------------
conversation_history = []

# -----------------------
# Embedding model
# -----------------------
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-large-en-v1.5"
)

# -----------------------
# Load vector index
# -----------------------
storage_context = StorageContext.from_defaults(persist_dir="./db")

index = load_index_from_storage(
    storage_context,
    embed_model=embed_model
)

vector_retriever = index.as_retriever(similarity_top_k=6)

bm25_retriever = BM25Retriever.from_defaults(
    nodes=list(index.docstore.docs.values()),
    similarity_top_k=6
)

reranker = CrossEncoder("BAAI/bge-reranker-base")

# -----------------------
# Routes
# -----------------------

@app.get("/")
def root():
    return {"message": "UCSM guide bot is running"}

@app.get("/query/")
def query_docs(query: str):

    # ---- Hybrid retrieval ----
    vector_nodes = vector_retriever.retrieve(query)
    bm25_nodes = bm25_retriever.retrieve(query)

    nodes = {n.node_id: n for n in vector_nodes}
    for n in bm25_nodes:
        nodes[n.node_id] = n

    nodes = list(nodes.values())

    if not nodes:
        return StreamingResponse(
            iter(["I could not find this in UCSM documentation."]),
            media_type="text/plain"
        )

    # ---- Reranking ----
    pairs = [(query, n.text) for n in nodes]
    scores = reranker.predict(pairs)

    ranked = list(zip(nodes, scores))
    ranked.sort(key=lambda x: x[1], reverse=True)

    top_nodes = [n for n, _ in ranked[:5]]

    context = "\n\n".join(n.text for n in top_nodes)
    history = "\n".join(conversation_history)

    prompt = f"""
You are a UCSM CLI troubleshooting assistant.

Conversation so far:
{history}

Rules:
- Use ONLY the context below.
- Do NOT invent CLI syntax.
- If command is not present, say:
  "I could not find this in UCSM documentation."
- Show commands in Markdown fenced blocks.
- Explain steps clearly.

Context:
{context}

Question:
{query}
"""

    conversation_history.append(f"User: {query}")

    def token_generator():
        answer_accum = ""
        for token in stream_llm(prompt):
            answer_accum += token
            yield token

        conversation_history.append(f"Assistant: {answer_accum}")

        if len(conversation_history) > 12:
            conversation_history.pop(0)
            conversation_history.pop(0)

    return StreamingResponse(token_generator(), media_type="text/plain")
