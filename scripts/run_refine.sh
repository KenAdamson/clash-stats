#!/usr/bin/env bash
# Wrapper to run refinement in tmux (handles CLAUDECODE unset and quoting)
unset CLAUDECODE
cd /home/kenadamson/clash-stats
./scripts/refine_with_claude_code.sh "replays/ScreenRecording_03-06-2026 16-24-20_1" > /tmp/refine_claude_code.log 2>&1
echo "Script exited with code $?" >> /tmp/refine_claude_code.log
