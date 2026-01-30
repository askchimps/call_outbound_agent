FROM python:3.13-slim

WORKDIR /app

# Install system dependencies including GLib/GObject for livekit-rtc
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libglib2.0-0 \
    libgobject-2.0-0 \
    libgstreamer1.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY agent.py .
COPY server.py .
COPY baseprompt.txt .

# Download required model files
RUN python agent.py download-files

# Expose ports
# 8000 - FastAPI server
EXPOSE 8000

# Run both the agent and the API server
CMD ["sh", "-c", "python agent.py dev & python server.py"]

