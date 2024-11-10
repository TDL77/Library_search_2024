# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    pkg-config \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p /app/data \
    /app/logs \
    /app/cache \
    /app/models \
    /app/embeddings \
    /app/index

# Copy requirements first for better cache usage
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set permissions for application directories
RUN chmod -R 777 /app/data \
    /app/logs \
    /app/cache \
    /app/models \
    /app/embeddings \
    /app/index

# Expose ports
EXPOSE 8501 5000

# Create a non-root user and switch to it
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Set healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501 || exit 1

# Start Streamlit app (can be overridden by docker-compose)
CMD ["streamlit", "run", "app_chart.py", "--server.port=8501", "--server.address=0.0.0.0"]