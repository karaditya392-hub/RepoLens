"""RepoLens Phase 3: embed chunks and store them in Qdrant.

Connection comes from environment variables:
    QDRANT_URL      (default http://localhost:6333)
    QDRANT_API_KEY  (optional, for Qdrant Cloud)

Embeddings are computed locally with fastembed (BAAI/bge-small-en-v1.5,
384 dims) — no external API key needed. The model (~130 MB) is downloaded
once on first use and cached.
"""

import os
import re
import uuid

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from chunker import Chunk

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384

_client: QdrantClient | None = None
_embedder: TextEmbedding | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(
            url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
            api_key=os.environ.get("QDRANT_API_KEY") or None,
        )
    return _client


def get_embedder() -> TextEmbedding:
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(model_name=EMBED_MODEL)
    return _embedder


def collection_name(repo_url: str) -> str:
    """github.com/pypa/sampleproject -> repolens_pypa_sampleproject"""
    slug = re.sub(r"^https://github\.com/", "", repo_url.rstrip("/"))
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", slug).strip("_").lower()
    return f"repolens_{slug}"


def chunk_text(c: Chunk) -> str:
    """Text that gets embedded: path + name + docstring + code."""
    parts = [f"File: {c.file_path}", f"{c.kind}: {c.name}"]
    if c.docstring:
        parts.append(c.docstring)
    parts.append(c.code)
    return "\n".join(parts)


def store_chunks(repo_url: str, chunks: list[Chunk]) -> str:
    """Embed and upsert chunks into a per-repo collection. Returns collection name."""
    client = get_client()
    name = collection_name(repo_url)

    if client.collection_exists(name):
        client.delete_collection(name)  # re-index from scratch
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )

    texts = [chunk_text(c) for c in chunks]
    vectors = list(get_embedder().embed(texts))

    points = [
        PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{repo_url}#{c.file_path}#{c.start_line}")),
            vector=vec.tolist(),
            payload={
                "repo_url": repo_url,
                "name": c.name,
                "kind": c.kind,
                "file_path": c.file_path,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "code": c.code,
                "docstring": c.docstring,
            },
        )
        for c, vec in zip(chunks, vectors)
    ]
    client.upsert(collection_name=name, points=points, wait=True)
    return name
