# Multi-stage build for the AegisMem service (Production Phase P8).
# Stage 1 resolves dependencies with uv into a venv; stage 2 is a slim runtime.
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app

# Install deps first (cached) from the lockfile, then the project.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --group pg
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --group pg

FROM python:3.12-slim AS runtime
# Run as a non-root user; the runtime holds no build tooling.
RUN useradd --create-home --uid 10001 aegis
WORKDIR /app
COPY --from=build --chown=aegis:aegis /app /app
ENV PATH="/app/.venv/bin:$PATH" \
    AEGISMEM_ENVIRONMENT=prod \
    PYTHONUNBUFFERED=1
USER aegis
EXPOSE 8000

# Liveness for orchestrators; readiness is /readyz.
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz').status==200 else 1)"

ENTRYPOINT ["aegismem", "serve", "--host", "0.0.0.0", "--port", "8000"]
