FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    git openssh-client curl cron build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyPI proxy — nginx caching proxy on the LAN
# BuildKit can't resolve .lan hostnames even with --network=host,
# so we use the IP directly. Override with --build-arg if IP changes.
ARG PYPI_HOST=192.168.7.58:8081
ARG PIP_INDEX_URL=http://192.168.7.58:8081/simple/
ARG PIP_TRUSTED_HOST=192.168.7.58
ARG PYTORCH_INDEX=http://192.168.7.58:8081/whl/xpu

# Copy source and install in one step
COPY pyproject.toml .
COPY src/ /app/src/
RUN pip install --no-cache-dir torch --index-url ${PYTORCH_INDEX} \
    && pip install --no-cache-dir ".[ml]" \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create data directory for SQLite volume mount
RUN mkdir -p /app/data

# Cron schedule — Debian cron reads from /etc/cron.d/
COPY crontab /etc/cron.d/cr-tracker
RUN chmod 0644 /etc/cron.d/cr-tracker

COPY entrypoint.sh /app/entrypoint.sh
COPY publish_stats.sh /app/publish_stats.sh

VOLUME ["/app/data"]

ENTRYPOINT ["/app/entrypoint.sh"]
