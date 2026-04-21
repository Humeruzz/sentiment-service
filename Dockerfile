# Base image — slim Python to keep image small
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install dependencies first (layer caching — only rebuilds if requirements change)
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Copy source code
COPY src/ ./src/

# Create directories for data and models, then drop root privileges
RUN mkdir -p /app/data /app/models \
    && useradd -m appuser \
    && chown -R appuser /app

USER appuser

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
