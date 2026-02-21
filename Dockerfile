FROM python:3.11-alpine

RUN apk add --no-cache git openssh-client curl

WORKDIR /app

# Copy source and install in one step
COPY pyproject.toml .
COPY src/ /app/src/
RUN pip install --no-cache-dir .

# Create data directory for SQLite volume mount
RUN mkdir -p /app/data

# Cron schedule — Alpine's crond is built into BusyBox, already present
COPY crontab /etc/crontabs/root

COPY entrypoint.sh /app/entrypoint.sh
COPY publish_stats.sh /app/publish_stats.sh

VOLUME ["/app/data"]

ENTRYPOINT ["/app/entrypoint.sh"]
