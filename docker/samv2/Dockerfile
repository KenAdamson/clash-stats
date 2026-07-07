# SAMv2 tracking sidecar for Intel Arc GPU (A770)
# Native torch XPU + current NEO compute runtime, same stack as cr-tracker.
# (Previously built on intel/intel-extension-for-pytorch:2.6.10-xpu — Intel
# stopped publishing IPEX images after torch 2.8; XPU support is upstream now.)
FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gnupg \
    && rm -rf /var/lib/apt/lists/*

# Intel GPU compute runtime (Level Zero + OpenCL). This sidecar is the canary
# for NEO updates before cr-tracker picks them up.
# NOTE: libze-intel-gpu1/libze1 are the current package names; the legacy
# intel-level-zero-gpu/level-zero pair is frozen at NEO 24.39 and its libigc1
# dependency conflicts with the libigc2 that current intel-opencl-icd needs.
# PINNED to 25.13: NEO 25.18 enumerates ZERO devices on small-BAR Arc systems
# ("Small BAR detected" warning, no ReBAR on this host) — regression tracked
# at https://github.com/intel/compute-runtime/issues/831. Re-test on the next
# repo release past 25.18 before bumping.
RUN curl -fsSL https://repositories.intel.com/gpu/intel-graphics.key | \
    gpg --dearmor -o /usr/share/keyrings/intel-graphics.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy unified" \
    > /etc/apt/sources.list.d/intel-gpu.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    libze-intel-gpu1=25.13.33276.22-1133~22.04 \
    intel-opencl-icd=25.13.33276.22-1133~22.04 \
    libze1 \
    && rm -rf /var/lib/apt/lists/*

# torch XPU wheels come from PyTorch's official index (not published to PyPI)
ARG PYTORCH_INDEX=https://download.pytorch.org/whl/xpu
RUN pip install --no-cache-dir torch --index-url ${PYTORCH_INDEX}

# SAMv2 + API dependencies. sam2 ships as sdist; --no-build-isolation makes
# its build use the torch-xpu installed above instead of pulling the multi-GB
# CUDA torch from PyPI into an isolated build env.
RUN pip install --no-cache-dir \
    setuptools \
    fastapi \
    "uvicorn[standard]" \
    pillow \
    numpy && \
    pip install --no-cache-dir --no-build-isolation sam2

# Download SAM2.1 checkpoints: Large (~860MB, validated default) and Small
# (~180MB, 1.74x faster on XPU — selected via SAMV2_MODEL_SIZE=small once
# validated on real game footage)
RUN mkdir -p /app/checkpoints && \
    python3 -c "import urllib.request; urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt', '/app/checkpoints/sam2.1_hiera_large.pt')" && \
    python3 -c "import urllib.request; urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt', '/app/checkpoints/sam2.1_hiera_small.pt')" && \
    ls -lh /app/checkpoints/

WORKDIR /app
COPY server.py /app/server.py

EXPOSE 8079

COPY start.py /app/start.py
ENTRYPOINT ["python3", "/app/start.py"]
