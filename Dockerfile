# CPU-only inference image for the frozen V1 model (plan §22).
FROM python:3.12-slim AS base

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv \
    PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 \
    TOKENIZER_PATH=/app/models/tokenizer_klue_roberta_base \
    MODEL_PATH=/app/models/tfidf_logistic.joblib

WORKDIR /app

# Dependencies first so the layer is cached across code changes.
# sys_platform == 'linux' selects the CPU-only torch index from pyproject.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --no-install-project

COPY src ./src
COPY app ./app
COPY models/tfidf_logistic.joblib ./models/tfidf_logistic.joblib
COPY models/tokenizer_klue_roberta_base ./models/tokenizer_klue_roberta_base

ENV PATH="/opt/venv/bin:$PATH"
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
