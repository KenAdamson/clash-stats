# Runbook: Beelink replica promotion — collation handling

**Status as of 2026-07-07.** The streaming standby at 192.168.7.62 warns:

> database "clash_stats" has a collation version mismatch
> The database was created using collation version 2.36, but the operating
> system provides version 2.41.

## Facts

- Primary (clash-postgres, P520): glibc provider, `en_US.utf8`,
  `datcollversion` 2.36 == runtime 2.36. **Self-consistent — do NOT run
  `REFRESH COLLATION VERSION` on the primary; there is nothing to refresh.**
- Standby (Beelink, LXC 102): OS glibc **2.41**. As a hot standby it only
  replays WAL — its indexes are byte-copies ordered by the *primary's* 2.36
  rules. Nothing can be fixed on it while it is read-only.
- Risk mechanism: btree text indexes assume the runtime locale's sort order.
  A glibc that sorts differently can make range scans AND equality probes
  miss rows — relevant because `DASHBOARD_DATABASE_URL` serves reads from
  this standby today.

## Empirical verification (2026-07-07)

Forced runtime sorts (`enable_indexscan/indexonlyscan/bitmapscan=off`) of the
three text-key domains, digest-compared across both nodes:

| domain | primary md5 (2.36) | replica md5 (2.41) |
|---|---|---|
| player_corpus.player_tag (ASCII `#`+base14) | `aa5edc11…` | identical |
| deck_cards.card_name (ASCII) | `766c0aa4…` | identical |
| player_corpus.player_name (full Unicode, 20K+) | `aa46d313…` | identical |

glibc 2.36 and 2.41 order our actual data identically → standby reads are
safe in practice **for current data**. Re-run this check after any Beelink
OS upgrade or if new heavily-Unicode data domains appear (the digest queries
are in this file's git history / memory).

## If the replica is PROMOTED (failover)

Immediately after promotion, **in this order**:

1. Accept the warning; the server runs fine while indexes are suspect.
2. Rebuild collation-dependent indexes **before trusting text lookups**:
   ```sql
   REINDEX DATABASE CONCURRENTLY clash_stats;  -- or, faster, targeted:
   -- REINDEX INDEX CONCURRENTLY battles_battle_id_key, battles_pkey,
   --   idx_battles_player_tag, idx_battles_opponent_tag,
   --   idx_battles_player_deck_hash, idx_replay_events_battle_id,
   --   replay_events_pkey, game_wp_summary_pkey, player_corpus_pkey;
   ```
   (Timestamp/int indexes are collation-immune; the digest evidence above
   suggests rebuilds will be no-ops for ordering, but rebuild anyway —
   evidence is point-in-time, promotion is forever.)
3. Only THEN record the new version:
   ```sql
   ALTER DATABASE clash_stats REFRESH COLLATION VERSION;
   ```
   (Refreshing *before* reindexing silences the warning without fixing
   anything — never do it first.)
4. Verify: reconnect; the mismatch warning must be gone.

## Permanent fix (recommended, Ken's call)

Run the standby's postgres from the **same container image as the primary**
(`tensorchord/vchord-postgres:pg17-v0.5.3`, Debian bookworm, glibc 2.36)
instead of the LXC's native packages. Same image ⇒ same glibc ⇒ standby is
bit-and-behavior identical, warning disappears legitimately, and promotions
need no reindex. Until then, this runbook is the compensating control.
