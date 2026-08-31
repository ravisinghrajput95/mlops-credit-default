# Multi-stage: the build stage carries uv and compiles dependencies; the runtime
# stage carries neither, keeping the published image small and its attack surface
# narrow. Image size also matters directly here -- Artifact Registry's always-free
# tier is 0.5 GB.

FROM python:3.12-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.1 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies are installed from the lockfile in their own layer so that editing
# application code does not invalidate the (slow) dependency install.
#
# --no-dev matters: `uv sync` installs the default dependency-group unless told
# otherwise, which put mypy, pytest, ruff and pre-commit in the runtime image.
# The uv version must also stay in step with the one that wrote uv.lock, since an
# older uv does not understand a newer lockfile revision.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra serve

COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --extra serve



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
