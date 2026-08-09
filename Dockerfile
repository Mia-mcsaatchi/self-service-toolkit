FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Bind to the platform-provided $PORT (Render, Cloud Run, Fly, …); fall back to
# 8000 for local `docker run`. Shell form so the env var expands.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]