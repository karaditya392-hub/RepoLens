FROM python:3.12-slim

# git is required at runtime: /index shallow-clones the target repo
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Keep resident memory inside the 512MB instance limit. glibc allocates a
# separate heap arena per thread (up to 8x CPU count), and the numeric stacks
# under ONNX Runtime each start their own thread pool. Left alone they push
# RSS past the cap and the instance is killed mid-request.
ENV MALLOC_ARENA_MAX=2 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image so cold starts don't re-download it
ENV FASTEMBED_CACHE_PATH=/app/model_cache
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

COPY chunker.py vectorstore.py rag.py app.py ./

ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
