FROM python:3.11-slim

WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install dependencies and system libs
RUN apt-get update && apt-get install -y \
    build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code and the local data directory into the image
COPY . .

# Expose backend ports (8000 for API)
EXPOSE 8000

ENV PYTHONUNBUFFERED=1

# Start FastAPI app
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
