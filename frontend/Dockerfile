# Use a lightweight Python image
FROM python:3.10-slim

# Set workdir
WORKDIR /app

# Copy requirements first (for better docker cache layer)
COPY requirements.txt .

# Install Python dependencies (streamlit, requests, etc.)
RUN pip install --no-cache-dir -r requirements.txt

# Copy all frontend code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Default command to run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
