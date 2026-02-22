#!/bin/bash
set -e

echo "=== cr-browser starting ==="

# Clean up stale locks from previous runs
rm -f /tmp/.X99-lock /tmp/.X11-unix/X99

# Start virtual framebuffer
Xvfb :99 -screen 0 1280x720x24 -ac &
XVFB_PID=$!
sleep 2

# Start VNC server (no password, shared mode)
x11vnc -display :99 -nopw -forever -shared -rfbport 5900 &
sleep 1

# Start noVNC web interface
websockify --web /usr/share/novnc 6080 localhost:5900 &

echo "  noVNC:      http://0.0.0.0:6080/vnc.html"
echo "  Playwright: ws://0.0.0.0:3000"
echo "=== cr-browser ready ==="

# Start Playwright server in foreground
exec npx playwright run-server --port 3000
