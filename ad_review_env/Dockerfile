FROM python:3.11-slim

WORKDIR /app

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv

# Copy all project files (build context = ad_review_env/)
COPY pyproject.toml .
COPY __init__.py .
COPY models.py .
COPY data.py .
COPY grader.py .
COPY agent.py .
COPY client.py .
COPY baseline.py .
COPY openenv.yaml .
COPY README.md .
COPY server/ ./server/

# Install dependencies (openenv-core provides the server framework)
RUN uv pip install --system -e ".[core]" || pip install openenv-core

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
