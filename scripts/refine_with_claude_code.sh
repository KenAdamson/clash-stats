#!/usr/bin/env bash
# Refine SAMv2 tracking labels using Claude Code's built-in vision.
# Run in tmux: tmux new -s refine './scripts/refine_with_claude_code.sh'
#
# Uses `claude -p` (print mode) — each frame is an independent invocation.
# No API key needed, uses Claude Code subscription.
#
# Usage:
#   ./scripts/refine_with_claude_code.sh <replay_dir> [start_frame]
#
# Example:
#   ./scripts/refine_with_claude_code.sh "replays/ScreenRecording_03-06-2026 16-24-20_1"

set -euo pipefail

REPLAY_DIR="${1:?Usage: $0 <replay_dir> [start_frame]}"
START_FRAME="${2:-0}"
LABEL_DIR="${REPLAY_DIR}/labels_samv2"
OUTPUT_DIR="${REPLAY_DIR}/labels_claude_code"

mkdir -p "$OUTPUT_DIR"

# Count total
TOTAL=$(find "$LABEL_DIR" -name 'label_*.json' | wc -l)
DONE=$(find "$OUTPUT_DIR" -name 'refined_*.json' | wc -l)
echo "Refining $TOTAL frames from $LABEL_DIR"
echo "Output: $OUTPUT_DIR"
echo "Already done: $DONE"
echo "---"

PROCESSED=0
FAILED=0
START_TIME=$(date +%s)

for LABEL_FILE in "$LABEL_DIR"/label_*.json; do
    FRAME_NUM=$(basename "$LABEL_FILE" | sed 's/label_//;s/\.json//')
    FRAME_NUM_INT=$((10#$FRAME_NUM))

    # Skip frames before start
    if [ "$FRAME_NUM_INT" -lt "$START_FRAME" ]; then
        continue
    fi

    OUTPUT_FILE="$OUTPUT_DIR/refined_${FRAME_NUM}.json"

    # Skip already refined
    if [ -f "$OUTPUT_FILE" ]; then
        continue
    fi

    FRAME_FILE="${REPLAY_DIR}/frame_${FRAME_NUM}.jpg"
    if [ ! -f "$FRAME_FILE" ]; then
        echo "WARN: No frame for $FRAME_NUM"
        continue
    fi

    # Read the label to extract metadata for the prompt
    LABEL_JSON=$(cat "$LABEL_FILE")
    GAME_TIME=$(echo "$LABEL_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['game_time_seconds'])")
    PERIOD=$(echo "$LABEL_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['period'])")
    PLAYER_DECK=$(echo "$LABEL_JSON" | python3 -c "import sys,json; print(', '.join(json.load(sys.stdin)['player_deck']))")
    OPPONENT_DECK=$(echo "$LABEL_JSON" | python3 -c "import sys,json; print(', '.join(json.load(sys.stdin)['opponent_deck']))")

    # Build units text
    UNITS_TEXT=$(echo "$LABEL_JSON" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for u in data.get('units', []):
    b = u['screen_bbox']
    print(f\"  - {u['team']} {u['card_name']} bbox=[{b[0]:.3f},{b[1]:.3f},{b[2]:.3f},{b[3]:.3f}] conf={u['confidence']:.0%} action={u['action']} +{u['time_since_play']:.1f}s\")
if not data.get('units'):
    print('  (none)')
")

    PROMPT="You are analyzing a Clash Royale gameplay frame. Read the image at ${FRAME_FILE} and analyze it.

VALID CARDS:
  Player deck: ${PLAYER_DECK}
  Opponent deck: ${OPPONENT_DECK}
  Valid sub-units: Skeleton (from Witch/Tombstone/Graveyard)

Game time: ${GAME_TIME}s | Period: ${PERIOD}

SAMv2 predicted units:
${UNITS_TEXT}

TASKS:
1. CONFIRM or REJECT each predicted unit — is it visible?
2. REFINE bounding boxes to tightly fit visible units (normalized 0-1 coords).
3. ADD Skeleton sub-units only (from Witch/Tombstone/Graveyard). No towers.
4. READ opponent elixir (purple bar, 0-10), opponent 4 hand cards (MUST be from opponent deck or \"Unknown\"), selected card.

RULES: No towers as units. card_name must be from decks above or \"Skeleton\". Hand cards from opponent deck or \"Unknown\".

Output ONLY this JSON (no markdown, no explanation):
{\"units\":[{\"card_name\":\"\",\"team\":\"\",\"bbox\":[x1,y1,x2,y2],\"confidence\":0.0,\"action\":\"\",\"status\":\"\",\"notes\":\"\"}],\"added_units\":[{\"card_name\":\"Skeleton\",\"team\":\"\",\"bbox\":[x1,y1,x2,y2],\"confidence\":0.0,\"action\":\"\",\"spawned_by\":\"\",\"notes\":\"\"}],\"rejected_predictions\":[],\"replay_signals\":{\"opponent_elixir\":0,\"opponent_hand\":[],\"opponent_selected_card\":null},\"frame_notes\":\"\"}"

    # Call claude in print mode (needs 2 turns: 1 to read image, 1 to respond)
    RESULT=$(claude -p "$PROMPT" --max-turns 3 2>/dev/null) || {
        echo "FAIL: frame $FRAME_NUM"
        FAILED=$((FAILED + 1))
        continue
    }

    # Extract JSON from response — Claude may include explanation text around it
    CLEAN_JSON=$(echo "$RESULT" | python3 -c "
import sys, json, re
text = sys.stdin.read().strip()

# Try to find JSON block in markdown fence
m = re.search(r'\`\`\`(?:json)?\s*\n(.*?)\n\`\`\`', text, re.DOTALL)
if m:
    text = m.group(1).strip()
else:
    # Try to find a JSON object directly
    m = re.search(r'(\{.*\})', text, re.DOTALL)
    if m:
        text = m.group(1).strip()

try:
    data = json.loads(text)
    # Normalize field names (Claude sometimes uses 'owner' instead of 'team')
    for u in data.get('units', []) + data.get('added_units', []):
        if 'owner' in u and 'team' not in u:
            u['team'] = 'friendly' if u['owner'] == 'player' else u.pop('owner')
            u.pop('owner', None)
        if 'source' in u and 'status' not in u:
            u['status'] = 'confirmed' if 'confirm' in u.get('source', '') else 'adjusted'
            u.pop('source', None)
    # Add metadata from label
    label = json.load(open('$LABEL_FILE'))
    data['frame_number'] = label['frame_number']
    data['game_time_seconds'] = label['game_time_seconds']
    data['period'] = label['period']
    data['battle_id'] = label['battle_id']
    data['player_deck'] = label.get('player_deck', [])
    data['opponent_deck'] = label.get('opponent_deck', [])
    print(json.dumps(data, indent=2))
except (json.JSONDecodeError, KeyError) as e:
    print(f'ERROR: {e}', file=sys.stderr)
    sys.exit(1)
") || {
        echo "PARSE FAIL: frame $FRAME_NUM"
        FAILED=$((FAILED + 1))
        continue
    }

    echo "$CLEAN_JSON" > "$OUTPUT_FILE"
    PROCESSED=$((PROCESSED + 1))

    # Progress every 10 frames
    if [ $((PROCESSED % 10)) -eq 0 ]; then
        NOW=$(date +%s)
        ELAPSED=$((NOW - START_TIME))
        RATE=$(echo "scale=1; $ELAPSED / $PROCESSED" | bc)
        REMAINING=$(echo "scale=0; ($TOTAL - $DONE - $PROCESSED) * $RATE / 1" | bc)
        echo "Progress: $PROCESSED processed, $FAILED failed | ${RATE}s/frame | ~${REMAINING}s remaining"
    fi
done

NOW=$(date +%s)
ELAPSED=$((NOW - START_TIME))
echo "=== COMPLETE ==="
echo "Processed: $PROCESSED | Failed: $FAILED | Time: ${ELAPSED}s"
echo "Output: $OUTPUT_DIR"
