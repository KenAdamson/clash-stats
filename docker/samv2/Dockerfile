# SAMv2 tracking sidecar for Intel Arc GPU (A770)
# Uses Intel Extension for PyTorch (IPEX) for XPU acceleration
FROM intel/intel-extension-for-pytorch:2.6.10-xpu

# SAMv2 + API dependencies
RUN pip install --no-cache-dir \
    sam2 \
    fastapi \
    "uvicorn[standard]" \
    pillow \
    numpy

# Download SAM2.1 Large checkpoint (~900MB)
RUN mkdir -p /app/checkpoints && \
    python3 -c "import urllib.request; urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt', '/app/checkpoints/sam2.1_hiera_large.pt')" && \
    ls -lh /app/checkpoints/

WORKDIR /app
COPY server.py /app/server.py

# Disable IPEX bfloat16 auto-promotion that breaks SAMv2 float32 inference
ENV IPEX_OPTIMIZE_TRANSFORMERS=0
ENV IPEX_FP32_MATH_MODE=FP32

EXPOSE 8079

COPY start.py /app/start.py
ENTRYPOINT ["python3", "/app/start.py"]
