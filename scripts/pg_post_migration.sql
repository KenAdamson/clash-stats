-- Post-pgloader manual indexes for PostgreSQL
-- Run after pgloader completes and row counts are verified.
--
-- Usage:
--   docker exec -i clash-postgres psql -U clash_stats clash_stats < scripts/pg_post_migration.sql

-- Manual indexes (were added outside Alembic on MariaDB)
CREATE INDEX IF NOT EXISTS idx_battles_opponent_tag
    ON battles(opponent_tag);

CREATE INDEX IF NOT EXISTS idx_battles_corpus_result
    ON battles(corpus, battle_type, result);

CREATE INDEX IF NOT EXISTS idx_battles_corpus_player_time
    ON battles(corpus, player_tag, battle_time);

CREATE INDEX IF NOT EXISTS idx_battles_replay_stale
    ON battles(replay_fetched, battle_type);

-- Update query planner statistics
ANALYZE battles;
ANALYZE deck_cards;
ANALYZE replay_events;
ANALYZE win_probability;

-- Stamp Alembic so it knows the schema is at revision 001
-- (pgloader creates tables directly, bypassing Alembic)
CREATE TABLE IF NOT EXISTS alembic_version (
    version_num VARCHAR(32) NOT NULL,
    CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
);
DELETE FROM alembic_version;
INSERT INTO alembic_version (version_num) VALUES ('001');
