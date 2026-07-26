# ADR-010: RoyaleAPI Scraping Egress Split (Direct vs VPN)

**Status:** Implemented
**Date:** 2026-07-26
**Depends on:** ADR-007 (Training Data Pipeline)

## Context

Replay scraping is the bottleneck for every downstream model — the WP estimator (ADR-004), embeddings (ADR-003), and the corpus pipeline (ADR-007) all consume replay events. RoyaleAPI sits behind Cloudflare, so every request needs a valid `cf_clearance` cookie alongside the `__royaleapi_session_v2` Discord login cookie.

**On 2026-06-08 the residential IP `24.17.189.124` was hard-banned by RoyaleAPI.** All scraping was moved behind a dedicated VPN sidecar (`cr-scraper-vpn`, gluetun/Mullvad, single Estonian exit) with a paired FlareSolverr (`cr-scraper-fs`) minting `cf_clearance` through the same exit. That kept the pipeline alive but imposed a hard volume ceiling: a single shared exit cannot carry corpus-scale replay traffic.

Two facts changed the calculus:

1. **The ban was pinned to the old IP.** After a router cut-over the modem re-negotiated a new WAN address (`98.232.126.170`). A single no-cookie probe returned `cf-mitigated: challenge` with a `Just a moment...` interstitial — an *ordinary* Cloudflare challenge. A hard ban returns `cf-mitigated: block` / "Error 1010". With a residential-minted `cf_clearance`, real fetches returned HTTP 200, logged in, full replay links.

2. **The VPN exit is measurably degraded.** ~600 403s/24h, with corpus replay runs yielding ~0 replays. Direct runs of the same job yielded ~50 replays per 5-minute cycle.

## Decision

**Corpus replays egress DIRECT from the host IP. Personal and alt replays stay on the VPN.**

This is a deliberate blast-radius allocation, not a throughput optimisation:

| | egress | rationale |
|---|---|---|
| corpus replays | direct (residential) | high volume, and the data is **replaceable** — a re-ban costs throughput we can rebuild |
| personal + alt | VPN exit | low volume, and the data is **irreplaceable** — the CR API only exposes the last 25 battles, so a scraping outage loses personal history permanently |

If the residential IP is banned again, corpus throughput degrades to its previous state while personal history keeps flowing. The inverse assignment would risk the one dataset that cannot be re-collected.

### `cf_clearance` is bound to the egress IP

This is the constraint that shapes the implementation. A clearance token minted through the Mullvad exit is invalid from the residential IP and vice versa. Because all jobs previously shared `/app/data/royaleapi_session.json`, a mixed-egress setup would have the two paths continuously invalidating each other's cookie.

**The direct path therefore uses its own session file** (`royaleapi_session_direct.json`), seeded from the VPN session so the (non-IP-bound) Discord login cookie carries over, then re-minting its own `cf_clearance` via the standalone `flaresolverr` container — which egresses residentially, unlike `cr-scraper-fs`.

### Reactive 403 recovery

The same investigation exposed a latent defect. `cf_clearance` was only refreshed proactively, at pass start, when older than `CF_REFRESH_MAX_AGE` (1800s). Cloudflare's real trust window closes earlier than that under load, so the token died mid-pass and every remaining player in the run 403'd — observed as bursts of ~14-15 failures in a 16-player run.

The only recovery path was VPN rotation, which was **self-defeating**: rotating the exit changes the IP, which invalidates the IP-bound `cf_clearance`, and the retry reused the cookie header built *before* the rotation. On the direct path the branch was unreachable entirely (gated on `GLUETUN_CONTROL_URL`), so a 403 simply ended the pass.

**Recovery is now: rotate only if actually behind the VPN, then always re-mint `cf_clearance`, reload cookies and UA, rebuild the header, and retry.** Because the re-mint writes the shared session file, the players queued behind the failure pick up the fresh token — one burst costs a single ~4s refresh instead of a wiped run.

### Configuration

| Variable | Default | Meaning |
|---|---|---|
| `CORPUS_REPLAY_DIRECT` | `1` | `1` = corpus replays bypass the VPN; `0` = restore the previous all-VPN behaviour |
| `DIRECT_SESSION_PATH` | `/app/data/royaleapi_session_direct.json` | separate session for the direct path (own `cf_clearance`) |
| `DIRECT_FLARESOLVERR_URL` | `http://flaresolverr:8191/v1` | residential-egress solver |
| `GLUETUN_CONTROL_URL` | `http://cr-scraper-vpn:8000` | **must be set explicitly** — see trap below |

`entrypoint.sh` builds a separate export block for `corpus_replays.sh`; every other wrapper keeps the shared VPN block.

> **Trap:** `replay_http.py` defaults `GLUETUN_CONTROL_URL` to the VPN control URL when unset, and `entrypoint.sh` only emits the export when non-empty. A direct-egress job that merely *omits* the variable therefore still takes the rotation branch on a 403 — burning 60s waiting for a new exit **and rotating the shared VPN exit, invalidating personal/alt's `cf_clearance`**. Direct paths must export `GLUETUN_CONTROL_URL=""` explicitly. This is the same failure mode as `ROYALEAPI_PROXY`, and it is why the variable is now pinned in `docker-compose.yml` rather than left to the code default.

## Consequences

**Positive**
- Corpus replay yield went from ~0 to ~50 replays per 5-minute cycle.
- The single-exit volume ceiling no longer constrains corpus replay scale.
- 403s are now recoverable in ~4s instead of costing a whole pass, on **both** paths — the VPN route is a genuinely usable fallback again.
- `CORPUS_REPLAY_DIRECT=0` is a one-variable rollback.

**Negative / risks**
- The residential IP is exposed to RoyaleAPI again. The 2026-06 ban followed sustained corpus-scale volume, so rate limits still matter: corpus stays throttled at `CORPUS_REPLAY_RATE=1.0` req/s. Raising it is the main way to get re-banned.
- Two session files mean two `cf_clearance` lifecycles; the sliding Discord login cookie is renewed independently in each and could drift if one path goes idle for ~7 days.
- The direct path has no rotation escape hatch by design — if the residential IP is challenged persistently, the fallback is `CORPUS_REPLAY_DIRECT=0`, not an automatic recovery.

## Verification

- Ban status: single no-cookie probe → `cf-mitigated: challenge`, not `block`.
- Direct fetch: HTTP 200, 181KB page, 15 replay links via the real parser, `logout` marker present (session authenticated).
- Live cron: `Combined scrape: 16 players, 1 battles, 50 replays`, 0 corpus 403s after the cut-over vs 15 in the equivalent pre-cut-over window.
- 403 recovery: forced via monkeypatched `_http_get` returning 403 — confirmed `refresh_cf_clearance` fires and the request is retried; separately confirmed the refresh actually rotates the token value.
