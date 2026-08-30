# Multi-stage: the build stage carries uv and compiles dependencies; the runtime
# stage carries neither, keeping the published image small and its attack surface
# narrow. Image size also matters directly here -- Artifact Registry's always-free
# tier is 0.5 GB.

FROM python:3.12-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies are installed from the lockfile in their own layer so that editing
# application code does not invalidate the (slow) dependency install.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --extra serve

COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable --extra serve

# xgboost's default PyPI wheel bundles ~290 MB of NVIDIA CUDA libraries that a
# Cloud Run CPU instance can never use. Swapping in the official CPU-only build
# roughly halves the image and keeps it inside Artifact Registry's 0.5 GB free
# tier. It is done here rather than in pyproject.toml because training still
# wants the GPU-capable package, and the two cannot be installed together.
# Uninstalling xgboost leaves its CUDA dependencies orphaned, so they are removed
# explicitly -- that directory alone is ~290 MB.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip uninstall --python /app/.venv/bin/python xgboost nvidia-nccl-cu13 \
    && uv pip install --python /app/.venv/bin/python xgboost-cpu \
    && rm -rf /app/.venv/lib/python3.12/site-packages/nvidia


FROM python:3.12-slim-bookworm AS runtime

# libgomp is required by xgboost's OpenMP runtime; curl is used by the healthcheck.
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 appuser

WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --chown=appuser:appuser src/ ./src/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_SOURCE=registry \
    API_PORT=8000

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Cloud Run injects $PORT; default to 8000 everywhere else.
CMD ["sh", "-c", "uvicorn credit_default.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
