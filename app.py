"""RepoLens Phase 2: FastAPI service exposing repo ingestion.

Run locally:
    uvicorn app:app --reload --port 8000
"""

import re
import subprocess

from dotenv import load_dotenv

load_dotenv()  # QDRANT_URL, QDRANT_API_KEY from repolens/.env

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from chunker import extract_chunks
from rag import answer_question
from vectorstore import store_chunks

app = FastAPI(title="RepoLens", version="0.5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to the Vercel domain after Phase 6
    allow_methods=["*"],
    allow_headers=["*"],
)

GITHUB_URL_RE = re.compile(
    r"^https://github\.com/[\w.-]+/[\w.-]+/?$"
)

MAX_CHUNKS = 3000  # keep indexing manageable on a small instance


class IndexRequest(BaseModel):
    repo_url: str


class IndexResponse(BaseModel):
    status: str
    repo_url: str
    chunks: int
    files: int
    collection: str


class QueryRequest(BaseModel):
    repo_url: str
    question: str
    top_k: int = 5


class Source(BaseModel):
    file_path: str
    start_line: int
    end_line: int
    name: str
    kind: str
    score: float
    code: str


class QueryResponse(BaseModel):
    status: str
    answer: str
    model: str
    sources: list[Source]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_repo(req: QueryRequest):
    url = req.repo_url.strip().removesuffix(".git")
    if not GITHUB_URL_RE.match(url):
        return JSONResponse(
            status_code=422,
            content={"status": "error", "detail": "Not a valid GitHub repo URL (expected https://github.com/<owner>/<repo>)"},
        )
    if not req.question.strip():
        return JSONResponse(
            status_code=422,
            content={"status": "error", "detail": "Question must not be empty"},
        )

    try:
        result = answer_question(url, req.question.strip(), top_k=min(max(req.top_k, 1), 20))
    except LookupError as e:
        return JSONResponse(status_code=404, content={"status": "error", "detail": str(e)})
    except Exception as e:
        return JSONResponse(status_code=502, content={"status": "error", "detail": f"Query failed: {e}"})

    return QueryResponse(status="ok", **result)


@app.post("/index", response_model=IndexResponse)
def index_repo(req: IndexRequest):
    url = req.repo_url.strip().removesuffix(".git")

    if not GITHUB_URL_RE.match(url):
        return JSONResponse(
            status_code=422,
            content={"status": "error", "detail": "Not a valid GitHub repo URL (expected https://github.com/<owner>/<repo>)"},
        )

    try:
        chunks = extract_chunks(url)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr or ""
        if "could not read Username" in stderr or "Repository not found" in stderr:
            detail = "Repository not found (or it is private)"
        else:
            detail = f"git clone failed: {stderr.strip().splitlines()[-1] if stderr.strip() else 'unknown error'}"
        return JSONResponse(
            status_code=502,
            content={"status": "error", "detail": detail},
        )

    if not chunks:
        return JSONResponse(
            status_code=422,
            content={"status": "error", "detail": "No indexable source files found in this repository"},
        )

    if len(chunks) > MAX_CHUNKS:
        return JSONResponse(
            status_code=413,
            content={"status": "error", "detail": f"Repository too large ({len(chunks)} chunks; limit is {MAX_CHUNKS}). Try a smaller repo."},
        )

    try:
        collection = store_chunks(url, chunks)
    except Exception as e:
        return JSONResponse(
            status_code=502,
            content={"status": "error", "detail": f"Vector store error: {e}"},
        )

    return IndexResponse(
        status="ok",
        repo_url=url,
        chunks=len(chunks),
        files=len({c.file_path for c in chunks}),
        collection=collection,
    )
