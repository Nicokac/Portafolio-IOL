# syntax=docker/dockerfile:1

FROM python:3.10-slim AS builder
ENV UV_USE_REQUIREMENTS=true
WORKDIR /app
# Install only production dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.10-slim
ENV UV_USE_REQUIREMENTS=true
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/local /usr/local
COPY app.py config.json application controllers domain infrastructure services shared ui scripts ./
RUN useradd -m app && chown -R app:app /app && chmod +x scripts/start.sh
USER app
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 CMD curl -f http://localhost:8501/_stcore/health || exit 1
CMD ["./scripts/start.sh"]
