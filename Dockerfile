FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir ".[web]"

RUN mkdir -p /data
VOLUME ["/data"]

ENV EFFECTIVE_MEMORY_DB=/data/effective_memory.db

EXPOSE 8000

ENTRYPOINT ["emem", "serve", "--host", "0.0.0.0", "--port", "8000"]
