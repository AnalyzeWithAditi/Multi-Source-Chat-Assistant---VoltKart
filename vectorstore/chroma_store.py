"""Chroma vector store (local or OpenAI embeddings) + persisted uploaded-file registry."""
from __future__ import annotations

import json
import os
from datetime import datetime

from rules.file_rules import data_dir

from langchain_core.embeddings import Embeddings

EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


class LocalEmbeddings(Embeddings):
    """Free local embeddings (all-MiniLM-L6-v2 via ONNX, bundled with chromadb). Model downloads once (~80 MB)."""

    def __init__(self):
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        self.fn = DefaultEmbeddingFunction()

    def embed_documents(self, texts):
        return [[float(x) for x in v] for v in self.fn(list(texts))]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def default_embeddings():
    """EMBEDDINGS=local (default) or openai."""
    if os.getenv("EMBEDDINGS", "local").lower() == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=EMBED_MODEL)
    return LocalEmbeddings()


class VectorStore:
    def __init__(self, embeddings=None, persist_dir=None):
        from langchain_chroma import Chroma

        embeddings = embeddings or default_embeddings()
        self.db = Chroma(
            collection_name="uploads",
            embedding_function=embeddings,
            persist_directory=str(persist_dir or data_dir() / "chroma"),
        )

    @staticmethod
    def where(**kv):
        items = [{k: v} for k, v in kv.items()]
        return items[0] if len(items) == 1 else {"$and": items}

    def add(self, chunks: list[dict], batch: int = 200):
        from langchain_core.documents import Document

        for i in range(0, len(chunks), batch):
            part = chunks[i:i + batch]
            self.db.add_documents(
                [Document(page_content=c["text"], metadata=c["meta"]) for c in part],
                ids=[f"{c['meta']['source_file']}::{i + j}" for j, c in enumerate(part)],
            )

    def delete_file(self, name: str):
        self.db.delete(where={"source_file": name})

    def search(self, query: str, k: int = 8, where: dict | None = None):
        return self.db.similarity_search(query, k=k, filter=where)


# ---- persisted registry of uploaded files ----
def _reg_path():
    return data_dir() / "registry.json"


def load_registry() -> list[dict]:
    p = _reg_path()
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def save_registry(entries: list[dict]):
    _reg_path().write_text(json.dumps(entries, indent=2), encoding="utf-8")


def upsert_entry(entry: dict):
    entry.setdefault("uploaded_at", datetime.now().isoformat(timespec="seconds"))
    reg = [e for e in load_registry() if e["name"] != entry["name"]]
    reg.append(entry)
    save_registry(reg)
