# syntax=docker/dockerfile:1

FROM python:3.10-slim AS builder
WORKDIR /app
# Install only production dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.10-slim
WORKDIR /app
COPY --from=builder /usr/local /usr/local
COPY app.py config.json application controllers domain infrastructure services shared ui ./
RUN useradd -m app && chown -R app:app /app
USER app
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
