#!/bin/sh
set -e

echo "=== cr-browser starting ==="

# Start virtual framebuffer
Xvfb :99 -screen 0 1280x720x24 -ac &
sleep 1

# Start VNC server (no password, shared mode)
x11vnc -display :99 -nopw -forever -shared -rfbport 5900 &
sleep 1

# Start noVNC web interface
websockify --web /usr/share/novnc 6080 localhost:5900 &

echo "  noVNC:      http://0.0.0.0:6080/vnc.html"
echo "  Playwright: ws://0.0.0.0:3000"
echo "=== cr-browser ready ==="

# Start Playwright server in foreground (headed mode uses DISPLAY)
exec npx playwright run-server --browser chromium --port 3000
