"""RepoLens Phase 4: retrieve relevant chunks from Qdrant and answer with an LLM.

Needs GROQ_API_KEY in the environment (loaded from .env by app.py).
"""

from groq import Groq

from vectorstore import collection_name, get_client, get_embedder

MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are RepoLens, a code assistant that answers questions about a \
GitHub repository using only the code excerpts provided in the user's message.

Rules:
- Ground every claim in the provided excerpts. If the excerpts don't contain the \
answer, say so plainly instead of guessing.
- When you reference code, cite it as `file_path:start_line-end_line` so the user \
can find it.
- Answer concisely in prose; include short code snippets only when they clarify."""

_groq: Groq | None = None


def get_groq() -> Groq:
    global _groq
    if _groq is None:
        _groq = Groq()  # reads GROQ_API_KEY from the environment
    return _groq


def retrieve(repo_url: str, question: str, top_k: int = 5) -> list[dict]:
    """Return the most relevant chunk payloads (with scores) for a question."""
    name = collection_name(repo_url)
    client = get_client()
    if not client.collection_exists(name):
        raise LookupError(f"Repository not indexed yet: {repo_url}")

    vector = list(get_embedder().embed([question]))[0].tolist()
    hits = client.query_points(collection_name=name, query=vector, limit=top_k, with_payload=True)
    return [{"score": p.score, **p.payload} for p in hits.points]


def build_context(hits: list[dict]) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        parts.append(
            f"[Excerpt {i}] {h['file_path']}:{h['start_line']}-{h['end_line']} "
            f"({h['kind']} {h['name']})\n```\n{h['code']}\n```"
        )
    return "\n\n".join(parts)


def answer_question(repo_url: str, question: str, top_k: int = 5) -> dict:
    hits = retrieve(repo_url, question, top_k)

    user_message = (
        f"Repository: {repo_url}\n\n"
        f"Code excerpts:\n\n{build_context(hits)}\n\n"
        f"Question: {question}"
    )

    response = get_groq().chat.completions.create(
        model=MODEL,
        max_tokens=2048,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    return {
        "answer": response.choices[0].message.content,
        "model": response.model,
        "sources": [
            {
                "file_path": h["file_path"],
                "start_line": h["start_line"],
                "end_line": h["end_line"],
                "name": h["name"],
                "kind": h["kind"],
                "score": round(h["score"], 3),
                "code": h["code"],
            }
            for h in hits
        ],
    }
