# Base image — slim Python to keep image small
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app
ENV PYTHONPATH=/app

# Install git for MLflow commit SHA tracking
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Install dependencies first (layer caching — only rebuilds if requirements change)
COPY requirements.txt ./
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2 && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Create directories for data and models, then drop root privileges
RUN mkdir -p /app/data /app/models \
    && useradd -m appuser \
    && chown -R appuser /app

USER appuser

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
