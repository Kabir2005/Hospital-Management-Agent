FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install CPU-only torch first (sentence-transformers pulls torch in; without this pip
# resolves the default CUDA wheels, which are 1GB+ and unnecessary on CPU-only hosts
# like Render's free tier, and blow past its build size/time limits).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Tavily MCP
RUN npm install -g tavily-mcp@0.1.4

# Copy application code
COPY . .

# Create necessary directories if they don't exist
RUN mkdir -p databases/Hospital_RAG_db
RUN mkdir -p kailash_info_store

# Make entrypoint script executable
RUN chmod +x /app/docker-entrypoint.sh

# Expose port for FastAPI (Hugging Face Spaces expects 7860; Railway/Render inject $PORT)
EXPOSE 7860

# Set entrypoint
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Command to run the application
CMD ["sh", "-c", "uvicorn api_setup:app_fastapi --host 0.0.0.0 --port ${PORT:-7860}"]