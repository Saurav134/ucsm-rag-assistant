#!/bin/sh

echo "Starting UCSM RAG Service..."

mkdir -p db
mkdir -p logs

# Run ingest ONLY if db is empty
if [ ! -f "./db/chroma.sqlite3" ]; then
  echo "Vector DB not found. Running ingestion..."
  python rag_service/ingest.py
else
  echo "Vector DB already exists. Skipping ingestion."
fi

echo "Starting FastAPI server..."
exec uvicorn rag_service.api:app --host 0.0.0.0 --port 8000
