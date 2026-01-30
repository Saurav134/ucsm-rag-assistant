#!/bin/bash
set -e

echo "Starting UCSM RAG Service..."

export DB_DIR="${DB_DIR:-/data/db}"
mkdir -p "$DB_DIR"

(
  set -e
  echo "Index worker started..."

  if [ -f "$DB_DIR/docstore.json" ]; then
    echo "Local index already present. Worker exiting."
    exit 0
  fi

  echo "Checking for matching prebuilt index in HF dataset repo..."
  python - << 'PY' | tee /tmp/index_status.txt
from rag_service.index_sync import try_download_index
ok, h = try_download_index()
print(f"INDEX_OK={ok}")
print(f"DOCS_HASH={h}")
PY

  if grep -q "^INDEX_OK=True$" /tmp/index_status.txt; then
    echo "Downloaded matching prebuilt index. Worker exiting."
    exit 0
  fi

  echo "No matching prebuilt index found. Running ingestion..."
  python rag_service/ingest.py

  echo "Uploading new index to private dataset repo..."
  DOCS_HASH=$(grep "^DOCS_HASH=" /tmp/index_status.txt | cut -d= -f2)
  python - << PY
from rag_service.index_sync import upload_index
upload_index("${DOCS_HASH}")
PY

  echo "Index worker completed."
) &

exec uvicorn rag_service.api:app --host 0.0.0.0 --port 7860 --log-level info