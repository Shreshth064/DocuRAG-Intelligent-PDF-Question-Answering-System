# syntax=docker/dockerfile:1

# Python 3.12 is the middle of the project's 3.11-3.14 CI matrix and the
# version with the most reliable prebuilt wheels for chromadb's tree
# (onnxruntime, pydantic-core). On 3.13/3.14 a missing wheel forces a
# source build, which needs a compiler and inflates the image. The
# -bookworm suffix pins the Debian release so the base cannot drift.
ARG PYTHON_TAG=3.12-slim-bookworm


# ---------------------------------------------------------------------
# Builder: resolve and install dependencies into a self-contained venv.
# Nothing from this stage reaches the runtime image except /opt/venv,
# so uv, apt build tooling and all download caches are left behind.
# ---------------------------------------------------------------------
FROM python:${PYTHON_TAG} AS builder

# uv is pinned to the same version used locally. Copying the static
# binary avoids installing (and then shipping) pip-installed tooling.
COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /usr/local/bin/uv

# Build-only system packages. Needed only if a dependency has no wheel
# for this platform and must be compiled; kept out of the final image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN uv venv "$VIRTUAL_ENV"

# Dependencies are installed before the source is copied, so editing
# application code does not invalidate this layer.
WORKDIR /app
COPY requirements.txt ./
# No --mount=type=cache here on purpose: cache mounts require BuildKit,
# and the classic builder (what `apt install docker.io` gives you without
# the docker-buildx plugin) fails outright on them. --no-cache keeps the
# layer from carrying uv's download cache.
RUN uv pip install --no-cache --python "$VIRTUAL_ENV/bin/python" \
    -r requirements.txt


# ---------------------------------------------------------------------
# Runtime: the default image. Ships the venv and the application only.
# ---------------------------------------------------------------------
FROM python:${PYTHON_TAG} AS runtime

# curl is the one package that cannot live in the builder: HEALTHCHECK
# runs in this stage. --no-install-recommends keeps it to a few MB.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY --from=builder /opt/venv /opt/venv

# Run as a non-root user. The persist directories are created and owned
# here so that named volumes mounted over them inherit this ownership
# instead of defaulting to root.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/ChromaDB /app/chroma_db \
    && chown -R appuser:appuser /app

WORKDIR /app
COPY --chown=appuser:appuser rag/ ./rag/
COPY --chown=appuser:appuser app.py main.py create_database.py ./

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# rag.api already binds 0.0.0.0:8000.
CMD ["python", "-m", "rag.api"]


# ---------------------------------------------------------------------
# Test: runtime plus the dev dependencies and the suite itself, so CI
# can reuse this image. Tests never land in the runtime stage above.
# ---------------------------------------------------------------------
FROM runtime AS test

USER root
COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /usr/local/bin/uv

COPY requirements.txt requirements-dev.txt ./
RUN uv pip install --no-cache --python "$VIRTUAL_ENV/bin/python" \
    -r requirements-dev.txt

COPY --chown=appuser:appuser pyproject.toml ./
COPY --chown=appuser:appuser tests/ ./tests/

USER appuser

HEALTHCHECK NONE
CMD ["pytest"]
