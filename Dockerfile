FROM python:3.11-alpine

WORKDIR /app

# Install dependencies first (layer caching)
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy application code
COPY src/ /app/src/

# Install the package itself
RUN pip install --no-cache-dir .

# Create data directory for SQLite volume mount
RUN mkdir -p /app/data

# Cron schedule — Alpine's crond is built into BusyBox, already present
COPY crontab /etc/crontabs/root

COPY entrypoint.sh /app/entrypoint.sh

VOLUME ["/app/data"]

ENTRYPOINT ["/app/entrypoint.sh"]
