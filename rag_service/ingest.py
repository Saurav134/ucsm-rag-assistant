import os
from dotenv import load_dotenv

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import chromadb

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv(override=True)

# ----------------------------
# Paths (Hugging Face Spaces persistent storage)
# ----------------------------
DB_DIR = os.environ.get("DB_DIR", "/data/db")
os.makedirs(DB_DIR, exist_ok=True)

# ----------------------------
# Load Documents
# ----------------------------
documents = SimpleDirectoryReader("./docs").load_data()

# ----------------------------
# Embedding Model (Open Source)
# ----------------------------
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-large-en-v1.5"
)

# ----------------------------
# Chroma Setup
# ----------------------------
chroma_client = chromadb.PersistentClient(path=DB_DIR)

collection_name = os.environ.get("CHROMA_COLLECTION", "dev_docs_collection")

try:
    chroma_collection = chroma_client.get_collection(collection_name)
except Exception:
    chroma_collection = chroma_client.create_collection(collection_name)

vector_store = ChromaVectorStore(chroma_collection)

# ----------------------------
# Build Index
# ----------------------------
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model,
    vector_store=vector_store
)

# ----------------------------
# Persist Index
# ----------------------------
index.storage_context.persist(persist_dir=DB_DIR)

print(f"Documents indexed successfully with BGE embeddings. Persisted to: {DB_DIR}")
