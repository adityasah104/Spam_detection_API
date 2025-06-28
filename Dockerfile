# Multi-stage build for smaller final image
FROM python:3.11.0 AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ cmake build-essential \
    libopencv-dev libdlib-dev libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip wheel
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11.0-slim

WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Set environment variables
ENV PYTHON_VERSION=3.11.0
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY . .

# Expose port
EXPOSE 8000
ENV PORT=8000

# Run the application
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:$PORT"]
