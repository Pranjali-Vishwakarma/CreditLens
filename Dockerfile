FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer-cached if requirements.txt unchanged)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Streamlit listens on 8501 by default
EXPOSE 8501

# Run preprocessing + app on container start.
# preprocess.py is idempotent: skips download/processing if files already exist.
CMD ["streamlit", "run", "src/app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
