#!/bin/bash
set -e

echo "=== cr-browser starting ==="

# Clean up stale locks from previous runs
rm -f /tmp/.X99-lock /tmp/.X11-unix/X99
rm -f /app/data/chromium-profile/SingletonLock /app/data/chromium-profile/SingletonSocket /app/data/chromium-profile/SingletonCookie

# Start virtual framebuffer
Xvfb :99 -screen 0 1280x720x24 -ac &
sleep 2

# Start VNC server (no password, shared mode)
x11vnc -display :99 -nopw -forever -shared -rfbport 5900 &
sleep 1

# Start noVNC web interface
websockify --web /usr/share/novnc 6080 localhost:5900 &

echo "  noVNC:      http://0.0.0.0:6080/vnc.html"
echo "  CDP:        http://0.0.0.0:9222 (via nginx)"

# Launch Chromium in headed mode with remote debugging
CHROMIUM=$(find /ms-playwright -name "chrome" -type f 2>/dev/null | head -1)
if [ -z "$CHROMIUM" ]; then
    CHROMIUM=$(which chromium || which google-chrome || which chromium-browser)
fi

echo "  Chromium:   $CHROMIUM"
echo "=== cr-browser ready ==="

"$CHROMIUM" \
    --no-sandbox \
    --disable-gpu \
    --disable-dev-shm-usage \
    --remote-debugging-port=9222 \
    --remote-allow-origins=* \
    --window-size=1280,720 \
    --user-data-dir=/app/data/chromium-profile \
    "about:blank" &
CHROME_PID=$!

# Wait for Chrome to start listening on 127.0.0.1:9222
sleep 2

# Nginx reverse proxy: exposes CDP on 0.0.0.0:9223, rewrites Host header
# so Chrome accepts the request. Also proxies WebSocket upgrades.
cat > /tmp/nginx-cdp.conf << 'NGINX'
daemon off;
error_log /dev/stderr;
events { worker_connections 64; }
http {
    access_log off;
    # Rewrite JSON responses: replace ws://localhost* URLs with
    # the correct address so Playwright connects back through nginx
    sub_filter_types application/json;
    sub_filter_once off;
    server {
        listen 9223;

        # WebSocket endpoints — proxy with upgrade
        location /devtools/ {
            proxy_pass http://127.0.0.1:9222;
            proxy_set_header Host localhost;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }

        # HTTP/JSON endpoints — rewrite webSocketDebuggerUrl
        location / {
            proxy_pass http://127.0.0.1:9222;
            proxy_set_header Host localhost;
            proxy_http_version 1.1;
            sub_filter 'ws://localhost/' 'ws://$host:9223/';
            sub_filter 'ws://localhost:9222/' 'ws://$host:9223/';
            sub_filter 'ws://127.0.0.1:9222/' 'ws://$host:9223/';
        }
    }
}
NGINX
nginx -c /tmp/nginx-cdp.conf &

echo "  CDP proxy:  0.0.0.0:9223 -> 127.0.0.1:9222 (nginx)"

# Wait for Chrome process (keeps container alive)
wait $CHROME_PID
