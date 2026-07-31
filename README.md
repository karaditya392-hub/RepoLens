# RepoLens — backend

Index a public GitHub repository into a vector store, then ask questions and get
answers grounded in the actual source, with file/line citations.

## How it works

1. `POST /index` shallow-clones the repo, filters to source files (skipping
   binaries, lockfiles, `node_modules`/`venv`/build output), and splits Python
   files into function/class/method chunks with the `ast` module. Other source
   files become whole-file chunks.
2. Each chunk is embedded locally with **fastembed** (`BAAI/bge-small-en-v1.5`,
   384-dim) — no embedding API key needed — and upserted into a per-repo
   **Qdrant** collection named `repolens_<owner>_<repo>`.
3. `POST /query` embeds the question, retrieves the top-k chunks from Qdrant,
   and sends them plus the question to **Groq** (`llama-3.3-70b-versatile`)
   with instructions to answer only from those excerpts and cite
   `file_path:start-end`.

## Endpoints

| Method | Path      | Body                                   | Returns |
| ------ | --------- | -------------------------------------- | ------- |
| GET    | `/health` | —                                      | `{"status": "ok"}` |
| POST   | `/index`  | `{"repo_url"}`                         | chunk count, file count, collection name |
| POST   | `/query`  | `{"repo_url", "question", "top_k"?}`   | answer, model, sources[] |

Error responses are `{"status": "error", "detail": "..."}` with status codes:
422 (invalid URL / empty question / nothing indexable), 404 (repo not indexed
yet), 413 (repo too large), 502 (clone, vector store, or LLM failure).

Interactive docs: `/docs`.

## Run locally

```bash
pip install -r requirements.txt
cp .env.example .env    # then fill in your values
uvicorn app:app --reload --port 8000
```

Requires `git` on PATH (used to clone target repos).

## Deploy to Render

The repo ships a `render.yaml` blueprint and a `Dockerfile` (Docker is used so
that `git` is available at runtime and the embedding model is baked into the
image, avoiding a slow first request).

1. Push this repo to GitHub.
2. In Render: **New → Blueprint**, select the repo, apply `render.yaml`.
3. Set the three environment variables when prompted (they are marked
   `sync: false`, so Render asks rather than reading them from the file):
   `QDRANT_URL`, `QDRANT_API_KEY`, `GROQ_API_KEY`.
4. Wait for the build, then check `https://<service>.onrender.com/health`.

Note: on Render's free tier the service sleeps after ~15 minutes idle, so the
first request after a gap takes roughly a minute to wake.
