"""Per-player play rate and poll cadence, for capacity-sized scheduling.

The corpus scrape has always swept players in last-scraped order, which gives
every player the same effective cadence (~10 days at current size). That is
close to right for the MEDIAN player, who plays 2.83 games/day and fills the
~30-battle API window in about 10.5 days -- but wrong for the tail. 46% of
measurable players need a faster cadence, and 25.4% of polls already come back
at the window cap, meaning those players outplayed the window before we arrived
and the overflow is gone permanently.

The fix needs a per-player number, which is what these columns hold:

  play_rate         games/day, measured from the SPAN of a player's most recent
                    captured battles. This is poll-independent and stays exact
                    even when the poll truncated: 30 battles spanning 6 hours is
                    120 games/day whether or not we missed earlier ones. That
                    property is what makes the whole scheme measurable from data
                    we already have, rather than needing new instrumentation.
  poll_cadence_days window_cap / play_rate -- how long until this player fills
                    the window and we start losing games.
  rate_measured_at  when the estimate was taken; a stale rate should decay back
                    toward the default rather than be trusted indefinitely.

Nullable throughout: a player with too few battles to measure has no rate, and
the scheduler must fall back to FIFO for them rather than inventing a number.

Revision ID: 014
Revises: 013
"""

from alembic import op
import sqlalchemy as sa

revision = "014"
down_revision = "013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("player_corpus", sa.Column("play_rate", sa.Float(), nullable=True))
    op.add_column("player_corpus", sa.Column("poll_cadence_days", sa.Float(), nullable=True))
    op.add_column("player_corpus",
                  sa.Column("rate_measured_at", sa.DateTime(), nullable=True))
    # The scheduler's hot query is "who is due?" -- active players ordered by how
    # far past their cadence they are. Without this it degrades to a seq scan of
    # the whole corpus every batch, once a minute.
    op.create_index("idx_player_corpus_due", "player_corpus",
                    ["active", "last_scraped"],
                    postgresql_where=sa.text("active = 1"))


def downgrade() -> None:
    op.drop_index("idx_player_corpus_due", table_name="player_corpus")
    op.drop_column("player_corpus", "rate_measured_at")
    op.drop_column("player_corpus", "poll_cadence_days")
    op.drop_column("player_corpus", "play_rate")
