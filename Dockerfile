FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir ".[web]"

RUN mkdir -p /data
VOLUME ["/data"]

ENV EFFECTIVE_MEMORY_DB=/data/effective_memory.db

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
  CMD python3 -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz', timeout=3).status == 200 else 1)"

ENTRYPOINT ["emem", "serve", "--host", "0.0.0.0", "--port", "8000"]
