"""Flask dashboard for Clash Royale battle analytics."""

import os
from pathlib import Path

# Must be set before prometheus_client is first imported
os.environ.setdefault("PROMETHEUS_DISABLE_CREATED_SERIES", "true")

from flask import Flask, Response, jsonify, render_template, request
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from tracker import analytics
from tracker.database import get_engine, get_session, init_db
from tracker.metrics import filter_in_process_metrics, render_accumulated_metrics

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

    engine = init_db(db_path)
    app.config["ENGINE"] = engine

    @app.route("/")
    def index():
        return render_template("dashboard.html")

    @app.route("/api/overview")
    def api_overview():
        session = get_session(engine)
        try:
            lo = _ladder_only()
            stats = analytics.get_overall_stats(session, ladder_only=lo)
            api_stats = analytics.get_all_time_api_stats(session)
            diff = analytics.get_snapshot_diff(session)
            return jsonify({
                "tracked": stats,
                "api_stats": api_stats,
                "snapshot_diff": diff,
            })
        finally:
            session.close()

    @app.route("/api/trophy-history")
    def api_trophy_history():
        session = get_session(engine)
        try:
            history = analytics.get_trophy_history(session, ladder_only=_ladder_only())
            return jsonify(history)
        finally:
            session.close()

    @app.route("/api/matchups")
    def api_matchups():
        session = get_session(engine)
        try:
            lo = _ladder_only()
            card_matchups = analytics.get_card_matchup_stats(session, min_battles=3, ladder_only=lo)
            archetypes = analytics.get_archetype_stats(session, min_battles=3, ladder_only=lo)
            return jsonify({
                "card_matchups": card_matchups,
                "archetypes": archetypes,
            })
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
        session = get_session(engine)
        try:
            lo = _ladder_only()
            streaks = analytics.get_streaks(session, ladder_only=lo)
            rolling_35 = analytics.get_rolling_stats(session, 35, ladder_only=lo)
            rolling_10 = analytics.get_rolling_stats(session, 10, ladder_only=lo)
            crowns = analytics.get_crown_distribution(session, ladder_only=lo)
            time_of_day = analytics.get_time_of_day_stats(session, ladder_only=lo)
            return jsonify({
                "streaks": streaks,
                "rolling_35": rolling_35,
                "rolling_10": rolling_10,
                "crown_distribution": crowns,
                "time_of_day": time_of_day,
            })
        finally:
            session.close()

    @app.route("/api/embeddings")
    def api_embeddings():
        """Return 3D embeddings for scatter plot visualization."""
        session = get_session(engine)
        try:
            from tracker.ml.storage import GameEmbedding, from_blob
            from tracker.models import Battle
            from sqlalchemy import select

            rows = session.execute(
                select(
                    GameEmbedding.battle_id,
                    GameEmbedding.embedding_3d,
                    GameEmbedding.cluster_id,
                )
            ).all()

            if not rows:
                return jsonify({"error": "No embeddings. Run --train-embeddings first."}), 404

            battle_ids = [r[0] for r in rows]
            battles = session.execute(
                select(Battle.battle_id, Battle.result, Battle.corpus, Battle.opponent_name)
                .where(Battle.battle_id.in_(battle_ids))
            ).all()
            meta = {b[0]: b for b in battles}

            points = []
            for bid, emb_3d, cluster_id in rows:
                try:
                    xyz = from_blob(emb_3d, 3)
                except ValueError:
                    continue  # Skip stale 2D embeddings pre-retraining
                b = meta.get(bid)
                points.append({
                    "battle_id": bid,
                    "x": float(xyz[0]),
                    "y": float(xyz[1]),
                    "z": float(xyz[2]),
                    "cluster_id": cluster_id,
                    "result": b[1] if b else None,
                    "corpus": b[2] if b else None,
                    "opponent": b[3] if b else None,
                })

            return jsonify({"points": points, "count": len(points)})
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
            })
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
