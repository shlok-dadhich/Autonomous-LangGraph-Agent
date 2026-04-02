FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	PIP_NO_CACHE_DIR=1 \
	SENTENCE_TRANSFORMERS_HOME=/opt/model-cache \
	VIRTUAL_ENV=/opt/venv \
	PATH="/opt/venv/bin:$PATH"

WORKDIR /app

RUN python -m venv "$VIRTUAL_ENV"

COPY requirements.txt ./

# Install CPU-only PyTorch first to avoid pulling CUDA builds and keep image lean.
RUN pip install --upgrade pip \
	&& pip install --index-url https://download.pytorch.org/whl/cpu torch \
	&& pip install sentence-transformers \
	&& grep -v -E "^(torch|sentence-transformers)$" requirements.txt > requirements.runtime.txt \
	&& pip install -r requirements.runtime.txt


FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	SENTENCE_TRANSFORMERS_HOME=/opt/model-cache \
	VIRTUAL_ENV=/opt/venv \
	PATH="/opt/venv/bin:$PATH"

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY . .

RUN python - <<'PY'
from sentence_transformers import SentenceTransformer

SentenceTransformer("all-MiniLM-L6-v2")
PY

# Persist DB files across container restarts/deployments.
VOLUME ["/data"]

# Keep app-compatible relative paths under /app/data while storing on volume.
RUN rm -rf /app/data && ln -s /data /app/data

CMD ["python", "main.py"]
