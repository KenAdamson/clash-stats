"""Flask dashboard for Clash Royale battle analytics."""

import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Must be set before prometheus_client is first imported
os.environ.setdefault("PROMETHEUS_DISABLE_CREATED_SERIES", "true")

from flask import Flask, Response, jsonify, render_template, request
from flask.json.provider import DefaultJSONProvider
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from tracker import analytics
from tracker.database import get_engine, get_session, init_db
from tracker.metrics import filter_in_process_metrics, render_accumulated_metrics


class _TTLCache:
    """Thread-safe in-memory cache with per-key TTL.

    Prevents duplicate expensive DB queries when the dashboard polls
    every 3 minutes and multiple browser tabs are open.
    """

    def __init__(self) -> None:
        self._store: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        """Return cached value if not expired, else None."""
        with self._lock:
            entry = self._store.get(key)
            if entry and time.monotonic() < entry[0]:
                return entry[1]
            return None

    def set(self, key: str, value: Any, ttl: float) -> None:
        """Cache value with TTL in seconds."""
        with self._lock:
            self._store[key] = (time.monotonic() + ttl, value)

    def clear(self) -> None:
        """Remove all cached entries."""
        with self._lock:
            self._store.clear()


_cache = _TTLCache()

TEMPLATE_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"


def _ladder_only() -> bool:
    """Check if the request should filter to ladder-only battles.

    Default is ladder-only. Pass ?mode=all to include everything.
    """
    return request.args.get("mode", "ladder") != "all"


def create_app(db_path: str | None = None) -> Flask:
    """Application factory for the dashboard.

    Args:
        db_path: Path to SQLite database. Falls back to CR_DB_PATH env var,
                 then default data/clash_royale_history.db.

    Returns:
        Configured Flask application.
    """
    app = Flask(
        __name__,
        template_folder=str(TEMPLATE_DIR),
        static_folder=str(STATIC_DIR),
    )

    if db_path is None:
        db_path = os.environ.get(
            "CR_DB_PATH", "data/clash_royale_history.db"
        )

    # Serialize datetime as ISO 8601 (not Flask's default RFC 2822)
    class _ISOJSONProvider(DefaultJSONProvider):
        @staticmethod
        def default(o: Any) -> Any:
            if isinstance(o, datetime):
                return o.strftime("%Y%m%dT%H%M%S.000Z")
            return DefaultJSONProvider.default(o)

    app.json_provider_class = _ISOJSONProvider
    app.json = _ISOJSONProvider(app)

    engine = init_db(db_path)
    app.config["ENGINE"] = engine

    # Clear stale cache entries from previous app instances (e.g. in tests)
    _cache.clear()

    @app.route("/")
    def index():
        return render_template("dashboard.html")

    # Cache TTL for dashboard endpoints (seconds).
    # Poll interval is 3 min — 90s TTL prevents duplicate queries
    # while staying reasonably fresh.
    CACHE_TTL = 90

    @app.route("/api/overview")
    def api_overview():
        cache_key = f"overview:{_ladder_only()}"
        cached = _cache.get(cache_key)
        if cached is not None:
            return jsonify(cached)
        session = get_session(engine)
        try:
            lo = _ladder_only()
            stats = analytics.get_overall_stats(session, ladder_only=lo)
            api_stats = analytics.get_all_time_api_stats(session)
            diff = analytics.get_snapshot_diff(session)
            result = {
                "tracked": stats,
                "api_stats": api_stats,
                "snapshot_diff": diff,
            }
            _cache.set(cache_key, result, CACHE_TTL)
            return jsonify(result)
        finally:
            session.close()

    @app.route("/api/trophy-history")
    def api_trophy_history():
        session = get_session(engine)
        try:
            history = analytics.get_trophy_history(session, ladder_only=_ladder_only())
            page = request.args.get("page", type=int)
            per_page = request.args.get("per_page", 1000, type=int)
            per_page = min(per_page, 5000)
            if page is not None:
                start = page * per_page
                chunk = history[start:start + per_page]
                return jsonify({
                    "data": chunk,
                    "page": page,
                    "total": len(history),
                    "has_more": start + per_page < len(history),
                })
            return jsonify(history)
        finally:
            session.close()

    @app.route("/api/matchups")
    def api_matchups():
        cache_key = f"matchups:{_ladder_only()}"
        cached = _cache.get(cache_key)
        if cached is not None:
            return jsonify(cached)
        session = get_session(engine)
        try:
            lo = _ladder_only()
            card_matchups = analytics.get_card_matchup_stats(session, min_battles=3, ladder_only=lo)
            archetypes = analytics.get_archetype_stats(session, min_battles=3, ladder_only=lo)
            result = {
                "card_matchups": card_matchups,
                "archetypes": archetypes,
            }
            _cache.set(cache_key, result, CACHE_TTL)
            return jsonify(result)
        finally:
            session.close()

    @app.route("/api/recent")
    def api_recent():
        session = get_session(engine)
        try:
            battles = analytics.get_recent_battles(session, limit=25, ladder_only=_ladder_only())
            return jsonify(battles)
        finally:
            session.close()

    @app.route("/api/simulation")
    def api_simulation():
        from tracker.simulation.runner import get_cached_results
        results = get_cached_results()
        if results is None:
            return jsonify({"error": "No simulation results yet. Run --sim-full first."}), 404
        return jsonify(results)

    @app.route("/api/replay-auth/start", methods=["POST"])
    def replay_auth_start():
        try:
            from tracker.replays import run_start_login
            run_start_login()
            return jsonify({"status": "navigated", "message": "Complete login via noVNC"})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/replay-auth/status")
    def replay_auth_status():
        try:
            from tracker.replays import run_check_login
            authenticated = run_check_login()
            return jsonify({"authenticated": authenticated})
        except Exception as e:
            return jsonify({"authenticated": False, "message": str(e)})

    @app.route("/api/streaks")
    def api_streaks():
        cache_key = f"streaks:{_ladder_only()}"
        cached = _cache.get(cache_key)
        if cached is not None:
            return jsonify(cached)
        session = get_session(engine)
        try:
            lo = _ladder_only()
            streaks = analytics.get_streaks(session, ladder_only=lo)
            rolling_35 = analytics.get_rolling_stats(session, 35, ladder_only=lo)
            rolling_10 = analytics.get_rolling_stats(session, 10, ladder_only=lo)
            crowns = analytics.get_crown_distribution(session, ladder_only=lo)
            time_of_day = analytics.get_time_of_day_stats(session, ladder_only=lo)
            corpus_traffic = analytics.get_corpus_traffic_by_hour(session)
            result = {
                "streaks": streaks,
                "rolling_35": rolling_35,
                "rolling_10": rolling_10,
                "crown_distribution": crowns,
                "time_of_day": time_of_day,
                "corpus_traffic": corpus_traffic,
            }
            _cache.set(cache_key, result, CACHE_TTL)
            return jsonify(result)
        finally:
            session.close()

    @app.route("/api/tilt")
    def api_tilt():
        """Return current tilt detection status."""
        cached = _cache.get("tilt")
        if cached is not None:
            return jsonify(cached)
        session = get_session(engine)
        try:
            from tracker.ml.tilt_detector import detect_tilt
            status = detect_tilt(session)
            result = {
                "level": status.level,
                "consecutive_losses": status.consecutive_losses,
                "recent_record": status.recent_record,
                "avg_leak": status.avg_leak_recent,
                "max_leak": status.max_leak_recent,
                "tilt_game_count": status.tilt_game_count,
                "embedding_matches": status.embedding_matches,
                "message": status.message,
            }
            _cache.set("tilt", result, CACHE_TTL)
            return jsonify(result)
        finally:
            session.close()

    @app.route("/api/wp/<battle_id>")
    def api_wp(battle_id: str):
        """Return P(win) curve data for a specific game."""
        session = get_session(engine)
        try:
            from tracker.ml.wp_storage import WinProbability, GameWPSummary

            rows = session.query(WinProbability).filter_by(
                battle_id=battle_id,
            ).order_by(WinProbability.game_tick).all()

            if not rows:
                return jsonify({"error": f"No WP data for {battle_id}"}), 404

            summary = session.query(GameWPSummary).filter_by(
                battle_id=battle_id,
            ).first()

            return jsonify({
                "battle_id": battle_id,
                "points": [
                    {
                        "tick": r.game_tick,
                        "win_prob": r.win_prob,
                        "wpa": r.wpa,
                        "criticality": r.criticality,
                        "event_index": r.event_index,
                    }
                    for r in rows
                ],
                "summary": {
                    "pre_game_wp": summary.pre_game_wp,
                    "final_wp": summary.final_wp,
                    "max_wp": summary.max_wp,
                    "min_wp": summary.min_wp,
                    "volatility": summary.volatility,
                    "top_positive_wpa_card": summary.top_positive_wpa_card,
                    "top_negative_wpa_card": summary.top_negative_wpa_card,
                    "critical_tick": summary.critical_tick,
                    "critical_card": summary.critical_card,
                } if summary else None,
            })
        finally:
            session.close()

    @app.route("/api/wp/cards")
    def api_wp_cards():
        """Return aggregate card WPA impact across personal games, split by side."""
        cached = _cache.get("wp_cards")
        if cached is not None:
            return jsonify(cached)
        session = get_session(engine)
        try:
            from tracker.ml.wp_storage import WinProbability, GameWPSummary
            from tracker.models import Battle, ReplayEvent
            from sqlalchemy import func, select, text as sa_text
            from collections import defaultdict

            # Get personal battle_ids with WP data
            personal_bids = session.execute(
                select(GameWPSummary.battle_id)
                .join(Battle, Battle.battle_id == GameWPSummary.battle_id)
                .where(Battle.corpus == "personal")
            ).scalars().all()

            if not personal_bids:
                return jsonify({"error": "No WP data. Run --wp-infer first."}), 404

            # Per-game card WPA by side — join WP with replay_events
            # Using numbered events to match event_index
            rows = session.execute(sa_text("""
                SELECT numbered.battle_id, numbered.card_name, numbered.side,
                       wp.wpa, wp.criticality
                FROM (
                    SELECT battle_id, card_name, side, game_tick,
                           ROW_NUMBER() OVER (
                               PARTITION BY battle_id ORDER BY game_tick, id
                           ) - 1 AS event_index
                    FROM replay_events
                    WHERE card_name != '_invalid'
                      AND battle_id IN :bids
                ) numbered
                JOIN win_probability wp
                  ON wp.battle_id = numbered.battle_id
                 AND wp.event_index = numbered.event_index
            """), {"bids": tuple(personal_bids)}).all()

            # Aggregate: per-game per-card-side cumulative WPA
            # Then count carry/liability/critical per card per side
            game_card_wpa = defaultdict(lambda: defaultdict(float))
            game_card_crit = defaultdict(lambda: defaultdict(float))

            for bid, card, side, wpa, crit in rows:
                key = (card, side)
                game_card_wpa[(bid, key)] = game_card_wpa.get((bid, key), 0) + float(wpa or 0)
                if abs(float(crit or 0)) > abs(game_card_crit.get((bid, key), 0)):
                    game_card_crit[(bid, key)] = float(crit or 0)

            # Per game: find top carry, liability, critical for each side
            carry = defaultdict(int)
            liability = defaultdict(int)
            critical = defaultdict(int)

            games_by_bid = defaultdict(dict)
            for (bid, key), total_wpa in game_card_wpa.items():
                games_by_bid[bid][key] = total_wpa

            for bid, card_wpas in games_by_bid.items():
                if not card_wpas:
                    continue
                best = max(card_wpas, key=card_wpas.get)
                worst = min(card_wpas, key=card_wpas.get)
                carry[best] += 1
                liability[worst] += 1

            # Critical: highest absolute criticality per game
            crit_by_game = defaultdict(lambda: (None, 0))
            for (bid, key), c in game_card_crit.items():
                if abs(c) > abs(crit_by_game[bid][1]):
                    crit_by_game[bid] = (key, c)
            for bid, (key, _) in crit_by_game.items():
                if key:
                    critical[key] += 1

            # Build card lists split by side
            all_keys = set(carry) | set(liability) | set(critical)
            team_cards = []
            opp_cards = []
            for key in all_keys:
                card, side = key
                c = carry.get(key, 0)
                l = liability.get(key, 0)
                cr = critical.get(key, 0)
                entry = {
                    "card": card,
                    "carry": c,
                    "liability": l,
                    "critical": cr,
                    "net": c - l,
                }
                if side == "team":
                    team_cards.append(entry)
                else:
                    opp_cards.append(entry)

            team_cards.sort(key=lambda x: x["net"], reverse=True)
            opp_cards.sort(key=lambda x: x["net"], reverse=True)

            # Average volatility
            summaries = session.execute(
                select(GameWPSummary.volatility)
                .join(Battle, Battle.battle_id == GameWPSummary.battle_id)
                .where(Battle.corpus == "personal")
            ).scalars().all()
            avg_vol = sum(v or 0 for v in summaries) / max(len(summaries), 1)

            result = {
                "total_games": len(personal_bids),
                "avg_volatility": round(avg_vol, 4),
                "team_cards": team_cards,
                "opp_cards": opp_cards,
            }
            _cache.set("wp_cards", result, CACHE_TTL)
            return jsonify(result)
        finally:
            session.close()

    @app.route("/api/wp/card/<card_name>")
    def api_wp_card_detail(card_name: str):
        """Return archetype breakdown for a specific card's WPA impact."""
        session = get_session(engine)
        try:
            import json as _json
            from tracker.ml.wp_storage import WinProbability
            from tracker.models import Battle, ReplayEvent
            from tracker.archetypes import classify_archetype
            from sqlalchemy import select, text as sa_text
            from collections import defaultdict

            # Get personal battles with WP data
            personal_battles = session.execute(
                select(Battle.battle_id, Battle.opponent_deck, Battle.result)
                .where(Battle.corpus == "personal")
                .where(Battle.battle_type == "PvP")
            ).all()
            bid_meta = {b[0]: (b[1], b[2]) for b in personal_battles}
            bids = list(bid_meta.keys())
            if not bids:
                return jsonify({"error": "No personal battles"}), 404

            # Get WPA for this card across all personal games
            rows = session.execute(sa_text("""
                SELECT numbered.battle_id, numbered.side, wp.wpa
                FROM (
                    SELECT battle_id, card_name, side, game_tick,
                           ROW_NUMBER() OVER (
                               PARTITION BY battle_id ORDER BY game_tick, id
                           ) - 1 AS event_index
                    FROM replay_events
                    WHERE card_name != '_invalid'
                      AND battle_id IN :bids
                ) numbered
                JOIN win_probability wp
                  ON wp.battle_id = numbered.battle_id
                 AND wp.event_index = numbered.event_index
                WHERE numbered.card_name = :card
            """), {"bids": tuple(bids), "card": card_name}).all()

            if not rows:
                return jsonify({"error": f"No WPA data for {card_name}"}), 404

            # Aggregate per-game WPA for this card
            game_wpa = defaultdict(float)
            game_side = {}
            for bid, side, wpa in rows:
                game_wpa[bid] += float(wpa or 0)
                game_side[bid] = side

            # Group by archetype
            archetype_data = defaultdict(lambda: {"games": 0, "total_wpa": 0.0, "wins": 0, "losses": 0})
            for bid, total_wpa in game_wpa.items():
                if bid not in bid_meta:
                    continue
                opp_deck_json, result = bid_meta[bid]
                try:
                    opp_deck = _json.loads(opp_deck_json) if opp_deck_json else []
                except (TypeError, _json.JSONDecodeError):
                    opp_deck = []
                archetype = classify_archetype(opp_deck)
                ad = archetype_data[archetype]
                ad["games"] += 1
                ad["total_wpa"] += total_wpa
                if result == "win":
                    ad["wins"] += 1
                elif result == "loss":
                    ad["losses"] += 1

            archetypes = []
            for arch, d in archetype_data.items():
                archetypes.append({
                    "archetype": arch,
                    "games": d["games"],
                    "avg_wpa": round(d["total_wpa"] / d["games"], 4) if d["games"] > 0 else 0,
                    "total_wpa": round(d["total_wpa"], 4),
                    "wins": d["wins"],
                    "losses": d["losses"],
                })

            archetypes.sort(key=lambda x: x["avg_wpa"], reverse=True)
            side = next(iter(game_side.values()), "unknown") if game_side else "unknown"

            return jsonify({
                "card": card_name,
                "side": side,
                "total_games": len(game_wpa),
                "archetypes": archetypes,
            })
        finally:
            session.close()

    CORPUS_SAMPLE_SIZE = 3000

    @app.route("/api/embeddings")
    def api_embeddings():
        """Return 3D embeddings for scatter plot visualization.

        Downsamples corpus points to ~3K for browser performance while
        keeping all personal games at full resolution.
        """
        # Longer TTL — embeddings change only on retrain
        cached = _cache.get("embeddings")
        if cached is not None:
            page = request.args.get("page", type=int)
            if page is None:
                return jsonify(cached)
        import random as _random
        session = get_session(engine)
        try:
            from tracker.ml.storage import GameEmbedding
            from tracker.models import Battle
            from sqlalchemy import select, func

            _emb_cols = (
                GameEmbedding.battle_id,
                GameEmbedding.embedding_vec_3d,
                GameEmbedding.cluster_id,
                Battle.result,
                Battle.corpus,
                Battle.opponent_name,
                Battle.battle_time,
            )
            _emb_join = select(*_emb_cols).join(
                Battle, GameEmbedding.battle_id == Battle.battle_id
            ).where(GameEmbedding.embedding_vec_3d.isnot(None))

            # All personal games (full resolution)
            personal_rows = session.execute(
                _emb_join.where(Battle.corpus == "personal")
            ).all()

            # Corpus games: random sample in SQL
            corpus_rows = session.execute(
                _emb_join.where(Battle.corpus != "personal")
                .order_by(func.random())
                .limit(CORPUS_SAMPLE_SIZE)
            ).all()

            if not personal_rows and not corpus_rows:
                return jsonify({"error": "No embeddings. Run --train-embeddings first."}), 404

            def _make_point(row):
                bid, vec_3d, cluster_id, result, corpus, opponent, battle_time = row
                if vec_3d is None or len(vec_3d) != 3:
                    return None
                return {
                    "battle_id": bid,
                    "x": float(vec_3d[0]),
                    "y": float(vec_3d[1]),
                    "z": float(vec_3d[2]),
                    "cluster_id": cluster_id,
                    "result": result,
                    "corpus": corpus,
                    "opponent": opponent,
                    "battle_time": battle_time,
                }

            personal_points = [p for p in (_make_point(r) for r in personal_rows) if p]
            corpus_points = [p for p in (_make_point(r) for r in corpus_rows) if p]
            total_corpus = session.execute(
                select(func.count()).select_from(GameEmbedding)
                .join(Battle, GameEmbedding.battle_id == Battle.battle_id)
                .where(Battle.corpus != "personal", GameEmbedding.embedding_vec_3d.isnot(None))
            ).scalar() or 0

            points = corpus_points + personal_points

            page = request.args.get("page", type=int)
            per_page = request.args.get("per_page", 1000, type=int)
            per_page = min(per_page, 5000)
            if page is not None:
                start = page * per_page
                chunk = points[start:start + per_page]
                return jsonify({
                    "points": chunk,
                    "page": page,
                    "total": len(points),
                    "total_corpus_full": total_corpus,
                    "has_more": start + per_page < len(points),
                })
            result = {
                "points": points,
                "count": len(points),
                "total_corpus_full": total_corpus,
            }
            _cache.set("embeddings", result, CACHE_TTL * 4)  # 6 min — changes rarely
            return jsonify(result)
        finally:
            session.close()

    @app.route("/api/similar/<battle_id>")
    def api_similar(battle_id: str):
        """Return top-10 most similar games, split by corpus/personal."""
        session = get_session(engine)
        try:
            from tracker.ml.similarity import find_similar
            results = find_similar(session, battle_id)
            if not results["corpus"] and not results["personal"]:
                return jsonify({"error": f"No embedding for {battle_id}"}), 404
            return jsonify({
                "reference": battle_id,
                "corpus": results["corpus"],
                "personal": results["personal"],
                "ref_coords": results.get("ref_coords"),
            })
        finally:
            session.close()

    @app.route("/api/manifold")
    def api_manifold():
        """Profile the 3-leg TCN embedding manifold."""
        session = get_session(engine)
        try:
            from tracker.ml.cluster_profiler import profile_manifold
            data = profile_manifold(session)
            if "error" in data:
                return jsonify(data), 404
            return jsonify(data)
        finally:
            session.close()

    @app.route("/api/temporal/matchup/<archetype>")
    def api_temporal_matchup(archetype: str):
        """Full temporal deep dive against an archetype."""
        session = get_session(engine)
        try:
            from tracker.temporal_analysis import matchup_deep_dive
            min_trophies = request.args.get("min_trophies", type=int)
            data = matchup_deep_dive(session, archetype, min_trophies=min_trophies)
            if "error" in data and data.get("game_count", -1) == 0:
                return jsonify(data), 404
            return jsonify(data)
        finally:
            session.close()

    @app.route("/api/temporal/opening")
    def api_temporal_opening():
        """Opening analysis (~30s) with optional filters."""
        session = get_session(engine)
        try:
            from tracker.temporal_analysis import opening_analysis
            archetype = request.args.get("archetype")
            min_trophies = request.args.get("min_trophies", type=int)
            data = opening_analysis(session, archetype=archetype, min_trophies=min_trophies)
            return jsonify(data)
        finally:
            session.close()

    @app.route("/api/temporal/phases")
    def api_temporal_phases():
        """Phase profile with optional filters."""
        session = get_session(engine)
        try:
            from tracker.temporal_analysis import phase_profile
            archetype = request.args.get("archetype")
            result = request.args.get("result")
            min_trophies = request.args.get("min_trophies", type=int)
            data = phase_profile(
                session, archetype=archetype, result=result,
                min_trophies=min_trophies,
            )
            return jsonify(data)
        finally:
            session.close()

    @app.route("/api/temporal/pushes")
    def api_temporal_pushes():
        """Push timing analysis with optional filters."""
        session = get_session(engine)
        try:
            from tracker.temporal_analysis import push_timing
            archetype = request.args.get("archetype")
            result = request.args.get("result")
            data = push_timing(session, archetype=archetype, result=result)
            return jsonify(data)
        finally:
            session.close()

    @app.route("/api/temporal/cycle")
    def api_temporal_cycle():
        """Broken cycle detection."""
        session = get_session(engine)
        try:
            from tracker.temporal_analysis import broken_cycle
            pairs_raw = request.args.get("pairs", "")
            window = request.args.get("window", 200, type=int)
            if not pairs_raw:
                return jsonify({"error": "pairs parameter required (e.g. miner:graveyard,witch:executioner)"}), 400
            pairs = []
            for spec in pairs_raw.split(","):
                parts = spec.strip().split(":")
                if len(parts) == 2:
                    pairs.append((parts[0], parts[1]))
            if not pairs:
                return jsonify({"error": "No valid pairs parsed"}), 400
            data = broken_cycle(session, pairs, window_ticks=window)
            return jsonify(data)
        finally:
            session.close()

    @app.route("/api/clusters")
    def api_clusters():
        """Return cluster profiles."""
        session = get_session(engine)
        try:
            from tracker.ml.clustering import profile_clusters
            profiles = profile_clusters(session)
            if not profiles:
                return jsonify({"error": "No clusters. Run --train-embeddings first."}), 404
            return jsonify({"clusters": profiles})
        finally:
            session.close()

    @app.route("/metrics")
    def metrics():
        """Serve Prometheus metrics.

        Combines in-process metrics from the dashboard with accumulated
        metrics from batch CLI jobs (corpus scrape, fetch, etc.).
        """
        in_process = filter_in_process_metrics(generate_latest().decode("utf-8"))
        # Ensure in-process metrics end with newline before appending batch metrics
        if in_process and not in_process.endswith("\n"):
            in_process += "\n"
        parts = [in_process]
        batch_metrics = render_accumulated_metrics()
        if batch_metrics:
            parts.append(batch_metrics)
        return Response("".join(parts), mimetype=CONTENT_TYPE_LATEST)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=8078, debug=False)
