FROM mcr.microsoft.com/playwright:v1.58.0-noble

RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb x11vnc novnc websockify \
    && rm -rf /var/lib/apt/lists/*

ENV DISPLAY=:99

EXPOSE 3000 6080

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
