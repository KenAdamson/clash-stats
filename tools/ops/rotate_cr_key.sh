#!/bin/bash
# Apply a re-issued Clash Royale API key with NO container restart.
#
# The CR developer portal binds each key to specific IPs. When the WAN address
# changes -- a modem re-negotiation, an ISP lease moving to a different pool --
# every call returns 403 and ALL ingest stops: personal, alt, corpus and the
# dimension refreshes alike. The battlelog only exposes the last 25 battles, so
# an outage longer than ~25 games loses them permanently. Recovery speed is the
# whole point of this script.
#
# entrypoint.sh bakes CR_API_KEY into each generated /app/*.sh cron wrapper at
# container start, so a new key in .env alone does nothing until a restart.
# Restarting is not free (it would abandon any in-flight training or backfill),
# so this rewrites the live wrappers in place AND updates .env for next boot.
#
# The key is read from STDIN, never from argv: a command-line argument is
# visible in /proc/<pid>/cmdline to every process on the box for as long as the
# command runs, and would land in shell history.
#
# Usage:
#   tools/ops/rotate_cr_key.sh              # prompts, reads from stdin
#   pbpaste | tools/ops/rotate_cr_key.sh    # or pipe it
set -euo pipefail

C=${CONTAINER:-cr-tracker}
ENV_FILE=${ENV_FILE:-$(dirname "$0")/../../.env}

if [ -t 0 ]; then
  printf 'Paste the new CR API key (input hidden), then Enter: ' >&2
  read -rs NEWKEY
  printf '\n' >&2
else
  read -r NEWKEY
fi

[ -n "${NEWKEY:-}" ] || { echo "no key supplied" >&2; exit 1; }
case "$NEWKEY" in
  eyJ*) ;;
  *) echo "that does not look like a JWT (expected to start 'eyJ')" >&2; exit 1;;
esac

# Show what the new key is bound to BEFORE installing it. Rotating to a key
# pinned to the wrong address just restarts the outage, and the payload says so
# for free.
echo "--- new key binding ---"
printf '%s' "$NEWKEY" | cut -d. -f2 | tr '_-' '/+' \
  | { base64 -d 2>/dev/null || true; } \
  | python3 -c "
import sys, json
try:
    d = json.loads(sys.stdin.read() or '{}')
except Exception:
    print('  (could not decode payload)'); raise SystemExit(0)
cidrs = [c for l in d.get('limits', []) if isinstance(l, dict) for c in (l.get('cidrs') or [])]
print('  cidrs:', cidrs or '(none listed)')
for c in cidrs:
    if '/' not in c:
        print('  WARNING: %s is a single address, not a range. A DHCP lease' % c)
        print('           change will break ingest again; prefer x.y.z.0/24.')
"
WAN=$(curl -s --max-time 15 https://api.ipify.org || echo "?")
echo "  current WAN: $WAN"

# Verify against the live API before touching anything. Installing a key that
# does not work would replace a diagnosed outage with an undiagnosed one.
echo "--- verifying against the CR API ---"
code=$(docker exec -e TESTKEY="$NEWKEY" "$C" sh -c \
  'curl -s -o /dev/null -w "%{http_code}" --max-time 20 \
   -H "Authorization: Bearer $TESTKEY" \
   "https://api.clashroyale.com/v1/players/%23L90009GPP"' || echo "000")
if [ "$code" != "200" ]; then
  echo "  FAILED: API returned $code — key NOT installed" >&2
  [ "$code" = "403" ] && echo "  403 means the key is bound to a different IP than $WAN" >&2
  exit 1
fi
echo "  OK: API returned 200"

# Rewrite every wrapper that carries the key. Done with python rather than sed
# so the key is passed via env and never appears in a command line, and so a
# key containing sed-special characters cannot corrupt the file.
mapfile -t FILES < <(docker exec "$C" sh -c 'grep -l CR_API_KEY /app/*.sh 2>/dev/null')
echo "--- rewriting ${#FILES[@]} wrappers ---"
for f in "${FILES[@]}"; do
  docker exec -e NEWKEY="$NEWKEY" -e TARGET="$f" "$C" python3 -c '
import os, re
p = os.environ["TARGET"]
s = open(p).read()
new = os.environ["NEWKEY"]
out, n = re.subn(r"(CR_API_KEY=)\"[^\"]*\"", lambda m: m.group(1) + "\"" + new + "\"", s)
if n == 0:
    out, n = re.subn(r"(CR_API_KEY=)[^\s\"]+", lambda m: m.group(1) + new, s)
open(p, "w").write(out)
print("  %s: %d occurrence(s)" % (p, n))
'
done

# .env is what the NEXT container start reads; skipping it means the fix silently
# reverts on the next recreate.
if [ -f "$ENV_FILE" ]; then
  NEWKEY="$NEWKEY" ENV_FILE="$ENV_FILE" python3 - <<'PY'
import os, re
p = os.environ["ENV_FILE"]
s = open(p).read()
out, n = re.subn(r"(?m)^CR_API_KEY=.*$", "CR_API_KEY=" + os.environ["NEWKEY"], s)
if n == 0:
    out = s.rstrip("\n") + "\nCR_API_KEY=" + os.environ["NEWKEY"] + "\n"
    n = 1
open(p, "w").write(out)
print("--- .env updated (%d line) ---" % n)
PY
else
  echo "WARNING: $ENV_FILE not found — the next container restart will revert to the OLD key" >&2
fi

echo "--- done. Crons pick this up on their next tick; no restart needed. ---"
echo "Watch recovery with:  docker logs -f --since 1m $C | grep -iE 'new battles|403'"
