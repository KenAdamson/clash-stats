"""Flask dashboard for Clash Royale battle analytics."""

import os
from pathlib import Path

from flask import Flask, jsonify, render_template

from tracker import analytics
from tracker.database import get_engine, get_session, init_db

TEMPLATE_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"


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
            stats = analytics.get_overall_stats(session)
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
            history = analytics.get_trophy_history(session)
            return jsonify(history)
        finally:
            session.close()

    @app.route("/api/matchups")
    def api_matchups():
        session = get_session(engine)
        try:
            card_matchups = analytics.get_card_matchup_stats(session, min_battles=3)
            archetypes = analytics.get_archetype_stats(session, min_battles=3)
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
            battles = analytics.get_recent_battles(session, limit=25)
            return jsonify(battles)
        finally:
            session.close()

    @app.route("/api/streaks")
    def api_streaks():
        session = get_session(engine)
        try:
            streaks = analytics.get_streaks(session)
            rolling_35 = analytics.get_rolling_stats(session, 35)
            rolling_10 = analytics.get_rolling_stats(session, 10)
            crowns = analytics.get_crown_distribution(session)
            time_of_day = analytics.get_time_of_day_stats(session)
            return jsonify({
                "streaks": streaks,
                "rolling_35": rolling_35,
                "rolling_10": rolling_10,
                "crown_distribution": crowns,
                "time_of_day": time_of_day,
            })
        finally:
            session.close()

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=8078, debug=False)
