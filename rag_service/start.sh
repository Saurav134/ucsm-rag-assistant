#!/bin/bash
set -e

echo "Starting UCSM RAG Service..."

# If DB folder is empty or missing, run ingestion
if [ ! -f "../db/chroma.sqlite3" ]; then
    echo "Vector DB not found. Running ingestion..."
    python ingest.py
else
    echo "Vector DB already exists. Skipping ingestion."
fi

echo "Starting FastAPI server..."
exec uvicorn api:app --host 0.0.0.0 --port 8000
