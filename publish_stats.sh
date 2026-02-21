#!/bin/sh
# Fetch dashboard API endpoints and force-push JSON files to the stats branch.
# Uses git plumbing so the working tree (main) is never disturbed.
# Produces a single orphan commit — no history accumulates.
set -e

REPO_DIR="${REPO_DIR:-/app}"
BRANCH="${STATS_BRANCH:-stats}"
REMOTE="${STATS_REMOTE:-origin}"
API_BASE="http://localhost:8078"

# If STATS_REPO_URL is set, use it as the push target instead of the named remote
PUSH_TARGET="${STATS_REPO_URL:-$REMOTE}"

cd "$REPO_DIR"

# Fetch all endpoints
overview=$(curl -sf "$API_BASE/api/overview")
trophy_history=$(curl -sf "$API_BASE/api/trophy-history")
matchups=$(curl -sf "$API_BASE/api/matchups")
recent=$(curl -sf "$API_BASE/api/recent")
streaks=$(curl -sf "$API_BASE/api/streaks")

if [ -z "$overview" ]; then
    echo "publish_stats: API not responding, skipping"
    exit 0
fi

# Build git tree from blobs (no checkout needed)
BLOB_OVERVIEW=$(echo "$overview" | git hash-object -w --stdin)
BLOB_TROPHY=$(echo "$trophy_history" | git hash-object -w --stdin)
BLOB_MATCHUPS=$(echo "$matchups" | git hash-object -w --stdin)
BLOB_RECENT=$(echo "$recent" | git hash-object -w --stdin)
BLOB_STREAKS=$(echo "$streaks" | git hash-object -w --stdin)

# Build stats/ subtree first, then wrap it in a root tree
SUBTREE=$(printf "100644 blob %s\toverview.json\n100644 blob %s\ttrophy-history.json\n100644 blob %s\tmatchups.json\n100644 blob %s\trecent.json\n100644 blob %s\tstreaks.json\n" \
    "$BLOB_OVERVIEW" "$BLOB_TROPHY" "$BLOB_MATCHUPS" "$BLOB_RECENT" "$BLOB_STREAKS" \
    | git mktree)

TREE=$(printf "040000 tree %s\tstats\n" "$SUBTREE" | git mktree)

COMMIT=$(echo "stats snapshot $(date -u +%Y-%m-%dT%H:%M:%SZ)" | git commit-tree "$TREE")

# Point the branch at this orphan commit and force-push
git update-ref "refs/heads/$BRANCH" "$COMMIT"
git push --force "$PUSH_TARGET" "$BRANCH" 2>&1

echo "publish_stats: pushed to $PUSH_TARGET $BRANCH at $(date -u +%H:%M:%SZ)"
