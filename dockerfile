
FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY requirements.txt ./

# Install CPU-only PyTorch to avoid unnecessary GPU/CUDA packages
RUN uv pip install --system --no-cache \
    --index-url https://download.pytorch.org/whl/cpu torch

# Install the remaining application dependencies
RUN uv pip install --system --no-cache -r requirements.txt

COPY . /app

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]