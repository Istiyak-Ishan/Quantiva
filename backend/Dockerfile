FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (Tesseract enables scanned-PDF OCR)
RUN apt-get update && apt-get install -y gcc curl tesseract-ocr tesseract-ocr-eng && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only PyTorch first to avoid downloading 2 GB of CUDA libs
RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

ENV HF_HOME=/opt/huggingface
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)"
RUN useradd --create-home --uid 10001 quantiva && chown -R quantiva:quantiva /app /opt/huggingface

COPY --chown=quantiva:quantiva . .
USER quantiva

# Expose port and provide default command
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --retries=5 CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/ready', timeout=4)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
