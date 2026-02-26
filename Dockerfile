FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    git openssh-client curl cron build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy source and install in one step
COPY pyproject.toml .
COPY src/ /app/src/
RUN pip install --no-cache-dir ".[ml]" \
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
