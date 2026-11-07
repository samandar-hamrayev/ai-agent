# syntax=docker/dockerfile:1

# ---- build: compile wheels in a throwaway stage ----
FROM python:3.12-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install -r requirements.txt

# ---- runtime: slim image, non-root user ----
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    CHROMADB_PATH=/data/chromadb \
    WORKSPACE_DIR=/data/workspace \
    HOME=/data

RUN useradd --create-home --uid 10001 agent \
    && mkdir -p /data/chromadb /data/workspace \
    && chown -R agent:agent /data
COPY --from=build /opt/venv /opt/venv
WORKDIR /app
COPY --chown=agent:agent . .

USER agent
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# Apply migrations (when a database is configured), then serve.
CMD ["sh", "-c", "if [ -n \"$DATABASE_URL\" ]; then alembic upgrade head; fi && exec uvicorn api.main:app --host 0.0.0.0 --port 8000 --proxy-headers"]
