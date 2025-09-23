FROM python:3.11-slim

ARG TOKEN="your_token"

# Update package list and install git
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages from GitHub
RUN pip install --no-cache-dir git+https://your_id:$TOKEN@github.com/umr-lops/asar-seastate-processor