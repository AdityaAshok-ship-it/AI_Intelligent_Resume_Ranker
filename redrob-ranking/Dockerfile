FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only torch first to avoid the ~2.5 GB CUDA wheel.
# pip will skip reinstalling when requirements.txt is processed next.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source (no large artifacts — sandbox embeds live)
COPY precompute.py rubric.py detectors.py reasoning.py rank.py ./
COPY sandbox/ ./sandbox/
COPY validate_submission.py ./

# Download bge-base-en-v1.5 weights into models/ at build time (~440 MB, one-time)
RUN python precompute.py --download-model --model-path models/bge-base-en-v1.5

EXPOSE 8501

CMD ["streamlit", "run", "sandbox/app.py", \
     "--server.address", "0.0.0.0", \
     "--server.port", "8501", \
     "--server.headless", "true"]
