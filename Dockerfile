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

# Python deps come from PyPI directly (pip's default). torch XPU wheels aren't
# published to PyPI, so they come from PyTorch's official XPU index. Override
# PYTORCH_INDEX with --build-arg only if mirroring wheels locally.
ARG PYTORCH_INDEX=https://download.pytorch.org/whl/xpu

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

# Create data directory for the volume mount
RUN mkdir -p /app/data

# Ops toolbox: training launchers, evaluation, diagnostics, feature research.
# These used to live loose in the ./data volume, which meant they were outside
# source control (data/ is a symlink git will not traverse) and had to be
# docker-cp'd in by hand. Baking them in keeps the image self-contained: the
# scripts that build and evaluate the models ship WITH the runtime that uses
# them. Placed after COPY src/ so editing a tool does not invalidate the
# multi-GB torch install layer.
COPY tools/ /app/tools/
RUN chmod -R a+rx /app/tools

# Cron schedule — Debian cron reads from /etc/cron.d/
#
# Ownership is load-bearing: cron SILENTLY refuses to execute any file in
# /etc/cron.d that is not owned by root. No error, no log line — every job in
# the file just stops, while cron itself keeps running and the file reads back
# perfectly. COPY already defaults to uid/gid 0, so the image is correct; the
# explicit chown is here to state the invariant and to survive any future
# change to COPY's defaults or a --chown added upstream.
#
# The trap is `docker cp`, which preserves the HOST file's ownership. Patching
# the live crontab that way cost 21.5 hours of ingest on 2026-08-06. Use
# tools/deploy_tracker.sh deploy_crontab(), which chowns and chmods.
COPY crontab /etc/cron.d/cr-tracker
RUN chown root:root /etc/cron.d/cr-tracker && chmod 0644 /etc/cron.d/cr-tracker

COPY entrypoint.sh /app/entrypoint.sh

VOLUME ["/app/data"]

ENTRYPOINT ["/app/entrypoint.sh"]
