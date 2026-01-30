import os
import hashlib
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download, HfApi

DOCS_DIR = os.environ.get("DOCS_DIR", "./docs")
DB_DIR = os.environ.get("DB_DIR", "/data/db")

REPO_ID = os.environ["INDEX_REPO_ID"]
REPO_TYPE = "dataset"

INDEX_ZIP = os.environ.get("INDEX_ZIP", "index.zip")
HASH_FILE = os.environ.get("HASH_FILE", "docs_hash.txt")


def compute_docs_hash() -> str:
    h = hashlib.sha256()
    base = Path(DOCS_DIR)
    files = sorted([p for p in base.rglob("*") if p.is_file()])
    for p in files:
        h.update(str(p.relative_to(base)).encode("utf-8"))
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def unzip_to_db(zip_path: str):
    os.makedirs(DB_DIR, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DB_DIR)


def zip_db(out_path: str):
    os.makedirs(DB_DIR, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(DB_DIR):
            for f in files:
                full = os.path.join(root, f)
                rel = os.path.relpath(full, DB_DIR)
                z.write(full, rel)


def try_download_index() -> tuple[bool, str]:
    local_hash = compute_docs_hash()

    try:
        remote_hash_path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            filename=HASH_FILE,
            token=os.environ.get("HF_TOKEN"),
        )
        remote_hash = Path(remote_hash_path).read_text().strip()
    except Exception:
        return (False, local_hash)

    if remote_hash != local_hash:
        return (False, local_hash)

    zip_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=INDEX_ZIP,
        token=os.environ.get("HF_TOKEN"),
    )
    unzip_to_db(zip_path)
    return (True, local_hash)


def upload_index(local_hash: str):
    api = HfApi(token=os.environ.get("HF_TOKEN"))

    tmp_zip = "/tmp/index.zip"
    tmp_hash = "/tmp/docs_hash.txt"

    zip_db(tmp_zip)
    Path(tmp_hash).write_text(local_hash)

    api.upload_file(
        path_or_fileobj=tmp_zip,
        path_in_repo=INDEX_ZIP,
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        commit_message="Update prebuilt index",
    )
    api.upload_file(
        path_or_fileobj=tmp_hash,
        path_in_repo=HASH_FILE,
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        commit_message="Update docs hash",
    )