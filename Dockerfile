FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    fastapi uvicorn[standard] httpx pillow numpy opencv-python-headless \
    pydantic onnx onnxruntime

# Clone SAM2 and install
RUN git clone https://github.com/facebookresearch/sam2.git /app/sam2-repo && \
    cd /app/sam2-repo && \
    pip install --no-cache-dir -e . && \
    cp -r sam2/configs /app/configs

# Download SAM2.1-tiny checkpoint
RUN mkdir -p /app/checkpoints && \
    wget -q -O /app/checkpoints/sam2.1_hiera_tiny.pt \
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"

RUN python -m pip install --no-cache-dir python-multipart

# Copy our code
COPY app.py /app/app.py
COPY export_onnx_decoder.py /app/export_onnx_decoder.py

# Export ONNX decoder at build time (one-time ~2 min)
RUN mkdir -p /app/onnx && \
    python /app/export_onnx_decoder.py || \
    echo "ONNX export failed — decoder-only mode unavailable, server fallback still works"

ENV SAM2_CHECKPOINT=/app/checkpoints/sam2.1_hiera_tiny.pt
ENV SAM2_CONFIG=configs/sam2.1/sam2.1_hiera_t.yaml
ENV ONNX_DECODER_PATH=/app/onnx/sam2.1_hiera_tiny_decoder.onnx
ENV ONNX_OUTPUT_DIR=/app/onnx
ENV MAX_CACHE_SIZE=100
ENV MAX_IMAGE_SIZE=1024
ENV LOG_LEVEL=INFO

EXPOSE 8100

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import httpx; r=httpx.get('http://localhost:8100/health'); assert r.status_code==200"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8100", "--workers", "1"]
