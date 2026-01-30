import os
import threading

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.retrievers.bm25 import BM25Retriever
from sentence_transformers import CrossEncoder

from rag_service.llm_client import stream_llm

app = FastAPI()

DB_DIR = os.environ.get("DB_DIR", "/data/db")
DOCSTORE_PATH = os.path.join(DB_DIR, "docstore.json")

RERANK_TOP_K = int(os.environ.get("RERANK_TOP_K", "5"))
RERANK_THRESHOLD = float(os.environ.get("RERANK_THRESHOLD", "0.15"))  # tune as needed

FALLBACK_MSG = "Hi! Ask me anything about Cisco UCS Manager CLI or GUI (Release 6.0). What are you trying to do?"

_init_lock = threading.Lock()

_embed_model = None
_index = None
_vector_retriever = None
_bm25_retriever = None
_reranker = None

# session_id -> history (store only user messages to avoid reinforcing bad assistant output)
_conversation_histories = {}
_hist_lock = threading.Lock()
_session_locks = {}


def _get_session_lock(session_id: str) -> threading.Lock:
    with _hist_lock:
        if session_id not in _session_locks:
            _session_locks[session_id] = threading.Lock()
        return _session_locks[session_id]


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        _embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")
    return _embed_model


def _get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("BAAI/bge-reranker-base")
    return _reranker


def _load_index():
    global _index, _vector_retriever, _bm25_retriever

    if not os.path.exists(DOCSTORE_PATH):
        return False

    storage_context = StorageContext.from_defaults(persist_dir=DB_DIR)
    _index = load_index_from_storage(storage_context, embed_model=_get_embed_model())

    _vector_retriever = _index.as_retriever(similarity_top_k=6)
    _bm25_retriever = BM25Retriever.from_defaults(
        nodes=list(_index.docstore.docs.values()),
        similarity_top_k=6,
    )
    return True


def _ensure_ready_or_503():
    global _index
    if _index is not None:
        return

    with _init_lock:
        if _index is not None:
            return

        ok = _load_index()
        if not ok:
            raise HTTPException(status_code=503, detail="Index is building. Try again shortly.")


def _fallback_stream():
    yield FALLBACK_MSG


@app.get("/")
def root():
    return {
        "message": "UCSM guide bot is running",
        "index_present": os.path.exists(DOCSTORE_PATH),
        "ready": _index is not None,
    }


@app.get("/warmup")
def warmup():
    _ensure_ready_or_503()
    _get_reranker()
    return {"ready": True}


@app.get("/query/")
def query_docs(query: str, session_id: str | None = None):
    _ensure_ready_or_503()

    q = (query or "").strip()
    if not q:
        return StreamingResponse(_fallback_stream(), media_type="text/plain")

    # simple greeting handling
    if q.lower() in {"hi", "hello", "hey", "hi!", "hello!", "hey!"}:
        return StreamingResponse(_fallback_stream(), media_type="text/plain")

    # ---- Hybrid retrieval ----
    vector_nodes = _vector_retriever.retrieve(q)
    bm25_nodes = _bm25_retriever.retrieve(q)

    nodes = {n.node_id: n for n in vector_nodes}
    for n in bm25_nodes:
        nodes[n.node_id] = n
    nodes = list(nodes.values())

    if not nodes:
        return StreamingResponse(_fallback_stream(), media_type="text/plain")

    # ---- Reranking ----
    reranker = _get_reranker()
    pairs = [(q, n.text) for n in nodes]
    scores = reranker.predict(pairs)

    ranked = list(zip(nodes, scores))
    ranked.sort(key=lambda x: x[1], reverse=True)

    top_ranked = ranked[:RERANK_TOP_K]
    top_nodes = [n for n, _ in top_ranked]

    # quality gate to reduce random answers
    best_score = float(top_ranked[0][1]) if top_ranked else -1.0
    if best_score < RERANK_THRESHOLD:
        return StreamingResponse(_fallback_stream(), media_type="text/plain")

    context = "\n\n".join(n.text for n in top_nodes)

    history_text = ""
    if session_id:
        session_lock = _get_session_lock(session_id)
        with session_lock:
            hist = _conversation_histories.get(session_id, [])
            history_text = "\n".join(hist[-6:])  # last few user turns only
            hist.append(f"User: {q}")
            _conversation_histories[session_id] = hist

    prompt = f"""
You are a UCSM troubleshooting assistant for both CLI and GUI.
Conversation so far:
{history_text}
Rules:
- Use ONLY the context below.
- Do NOT invent CLI command syntax or GUI steps.
- If the answer is not present in the context, respond exactly with:
  "{FALLBACK_MSG}"
- Show commands in Markdown fenced blocks when applicable.
- Explain steps clearly.
Context:
{context}
Question:
{q}
""".strip()

    def token_generator():
        for token in stream_llm(prompt):
            yield token

    return StreamingResponse(token_generator(), media_type="text/plain")