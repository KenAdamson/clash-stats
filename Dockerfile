FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    git openssh-client curl cron build-essential gnupg \
    && rm -rf /var/lib/apt/lists/*

# Intel GPU compute runtime (Level Zero) — required for torch XPU
RUN curl -fsSL https://repositories.intel.com/gpu/intel-graphics.key | \
    gpg --dearmor -o /usr/share/keyrings/intel-graphics.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy unified" \
    > /etc/apt/sources.list.d/intel-gpu.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    intel-level-zero-gpu level-zero \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyPI proxy — nginx caching proxy on the LAN
# BuildKit can't resolve .lan hostnames even with --network=host,
# so we use the IP directly. Override with --build-arg if IP changes.
ARG PYPI_HOST=192.168.7.58:8081
ARG PIP_INDEX_URL=http://192.168.7.58:8081/simple/
ARG PIP_TRUSTED_HOST=192.168.7.58
ARG PYTORCH_INDEX=http://192.168.7.58:8081/whl/xpu

# Layer 1: torch + OpenCL ICD (rarely changes, ~2GB, cached unless pyproject.toml changes)
# intel-opencl-icd must be installed alongside torch because torch XPU pins
# libigc1 while opencl-icd wants libigc2 — installing together lets apt resolve.
# Required for oneDNN's SDPA GPU primitive (Transformer attention on XPU).
COPY pyproject.toml .
RUN pip install --no-cache-dir torch --index-url ${PYTORCH_INDEX} && \
    apt-get update && apt-get install -y --no-install-recommends \
    intel-opencl-icd=24.39.31294.20-1032~22.04 \
    && rm -rf /var/lib/apt/lists/*

# Layer 2: remaining Python deps (cached unless pyproject.toml changes)
# Create minimal package structure so pip install works for deps only
RUN mkdir -p /app/src/tracker && \
    echo '__version__ = "0.0.0"' > /app/src/tracker/__init__.py && \
    pip install --no-cache-dir ".[ml]" && \
    apt-get purge -y --auto-remove build-essential && \
    rm -rf /var/lib/apt/lists/*

# Layer 3: actual source (changes often, but deps are cached above)
COPY src/ /app/src/
RUN pip install --no-cache-dir --no-deps .

# Create data directory for SQLite volume mount
RUN mkdir -p /app/data

# Cron schedule — Debian cron reads from /etc/cron.d/
COPY crontab /etc/cron.d/cr-tracker
RUN chmod 0644 /etc/cron.d/cr-tracker

COPY entrypoint.sh /app/entrypoint.sh
COPY publish_stats.sh /app/publish_stats.sh

VOLUME ["/app/data"]

ENTRYPOINT ["/app/entrypoint.sh"]
