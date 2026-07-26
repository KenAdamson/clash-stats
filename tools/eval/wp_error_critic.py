"""WP error-critic (Phase A).

Tests whether opponent-skill signal — which the WP model CANNOT see (its 51
features are placement/economy only) — explains the WP model's prediction
errors on personal games. Two critics:

  1. Outcome-lift: predict win/loss from WP-summary features alone (base) vs
     WP-summary + opponent-skill features (aug). If aug beats base on a
     time-held-out split, opponent skill carries signal WP is missing.
  2. Residual: regress WP over-confidence (max_wp - y) on opponent-skill
     features; importances name the culprit signal.

Read-only on the DB. sklearn only (HistGradientBoosting handles sparse NaNs).
"""
import os
import pickle
import json
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.inspection import permutation_importance

PLAYER = "#L90009GPP"
ENRICH = "/app/data/corpus_enrichment.pkl"
OUT = "/app/data/wp_error_critic_results.md"

e = create_engine(os.environ["DATABASE_URL"])

# --- assemble dataset ---
sql = text("""
    SELECT b.battle_id, b.battle_time, b.result,
           b.player_starting_trophies AS p_tr,
           b.opponent_starting_trophies AS o_tr,
           b.opponent_tag,
           b.opponent_elixir_leaked, b.player_elixir_leaked, b.battle_duration,
           s.pre_game_wp, s.final_wp, s.max_wp, s.min_wp, s.volatility,
           pd.latest_trophies AS o_latest_tr, pd.behavioral_gap AS o_beh_gap,
           pd.implied_trophy_gap AS o_impl_gap, pd.deck_top_level AS o_deck_lvl,
           pd.games AS o_h2h_games, pd.wins AS o_h2h_wins,
           pf.elixir_pace AS o_pace, pf.throughput AS o_thru, pf.reaction AS o_react,
           pf.pace_consistency AS o_pace_cons, pf.def_reaction AS o_def_react,
           pf.fast_react_frac AS o_fast, pf.latest_trophies AS o_fp_tr
    FROM battles b
    JOIN game_wp_summary s ON s.battle_id = b.battle_id
    LEFT JOIN player_dim pd ON pd.player_tag = b.opponent_tag
    LEFT JOIN pilot_fingerprint pf ON pf.player_tag = b.opponent_tag
    WHERE b.player_tag = :p AND b.battle_type = 'PvP'
      AND b.result IN ('win','loss') AND s.max_wp IS NOT NULL
    ORDER BY b.battle_time
""")
df = pd.read_sql(sql, e, params={"p": PLAYER})
df["y"] = (df["result"] == "win").astype(int)

# opponent efficiency + badge sheet from enrichment cache
enr = {}
if os.path.exists(ENRICH):
    with open(ENRICH, "rb") as f:
        enr = pickle.load(f)
def _eff(tag):
    v = enr.get(tag)
    if v and v.get("bc"):
        return v["best"] / v["bc"] if v.get("best") else np.nan
    return np.nan
def _badges(tag):
    v = enr.get(tag) or {}
    return v.get("n_badges", np.nan), (1.0 if v.get("has_years_played") else (0.0 if v else np.nan)), v.get("n_event_badges", np.nan)
df["o_eff"] = df["opponent_tag"].map(_eff)
bd = df["opponent_tag"].map(_badges)
df["o_nbadges"] = [b[0] for b in bd]
df["o_yp"] = [b[1] for b in bd]
df["o_events"] = [b[2] for b in bd]

# derived
df["trophy_gap"] = df["o_tr"] - df["p_tr"]
df["o_h2h_wr"] = df["o_h2h_wins"] / df["o_h2h_games"].replace(0, np.nan)

WP_FEATS = ["pre_game_wp", "final_wp", "max_wp", "min_wp", "volatility"]
# LEAKAGE-SAFE opponent features only. EXCLUDED: o_h2h_wr / wins / games —
# player_dim aggregates ALL battles incl. those in the test set, so H2H win
# rate encodes the outcome (drove a bogus AUC 0.999). Everything below is
# outcome-independent: trophies, card levels, behavioral timing, efficiency.
OPP_FEATS = ["o_tr", "trophy_gap", "o_latest_tr", "o_beh_gap", "o_impl_gap",
             "o_deck_lvl", "o_eff", "o_nbadges", "o_yp", "o_events",
             "o_pace", "o_thru", "o_react", "o_pace_cons", "o_def_react", "o_fast"]

n = len(df)
lines = []
def log(s): print(s); lines.append(s)

log(f"# WP Error-Critic Results\n")
log(f"Dataset: {n} personal PvP games with WP summary. Wins {df['y'].sum()} / Losses {(1-df['y']).sum()} (WR {df['y'].mean():.1%}).\n")
log("## Opponent-feature coverage (non-null fraction)")
for c in OPP_FEATS:
    log(f"- `{c}`: {df[c].notna().mean():.0%}")
log("")

# --- time-ordered split (last 20% val, mirroring WP training) ---
cut = int(n * 0.8)
tr, va = df.iloc[:cut], df.iloc[cut:]
log(f"Time split: train {len(tr)} (through {tr['battle_time'].max()}), val {len(va)} (from {va['battle_time'].min()}). Val WR {va['y'].mean():.1%}.\n")

def evaluate(feats, name):
    Xtr, Xva = tr[feats].values.astype(float), va[feats].values.astype(float)
    ytr, yva = tr["y"].values, va["y"].values
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                       max_depth=3, l2_regularization=1.0,
                                       random_state=0, validation_fraction=None)
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xva)[:, 1]
    auc = roc_auc_score(yva, p); ll = log_loss(yva, p, labels=[0, 1]); br = brier_score_loss(yva, p)
    log(f"- **{name}**: AUC {auc:.3f} | log-loss {ll:.3f} | Brier {br:.3f}  ({len(feats)} feats)")
    return m, p, auc, ll, br

log("## Critic 1 — outcome-prediction lift (val, time-held-out)")
_, p_base, auc_b, ll_b, br_b = evaluate(WP_FEATS, "Base (WP summary only)")
_, p_aug, auc_a, ll_a, br_a = evaluate(WP_FEATS + OPP_FEATS, "Aug (WP + opponent skill)")
_, p_opp, auc_o, ll_o, br_o = evaluate(OPP_FEATS, "Opponent-skill only")
log(f"\n**Lift (aug − base): AUC {auc_a-auc_b:+.3f} | log-loss {ll_a-ll_b:+.3f} | Brier {br_a-br_b:+.3f}**")
log("(negative log-loss/Brift = better; positive AUC = better)\n")

# also: how much does WP's own max_wp already predict outcome, vs opp-only
log(f"Reference: WP `max_wp` alone AUC {roc_auc_score(va['y'], va['max_wp']):.3f}; "
    f"WP `final_wp` alone AUC {roc_auc_score(va['y'], va['final_wp']):.3f}.\n")

# --- Critic 2: residual regression ---
log("## Critic 2 — residual critic (predict WP over-confidence)")
df["resid"] = df["max_wp"] - df["y"]   # >0 = overconfident (esp. lost games)
rtr, rva = df.iloc[:cut], df.iloc[cut:]
Xtr = rtr[OPP_FEATS].values.astype(float); Xva = rva[OPP_FEATS].values.astype(float)
reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=3,
                                    l2_regularization=1.0, random_state=0)
reg.fit(Xtr, rtr["resid"].values)
pv = reg.predict(Xva)
from sklearn.metrics import r2_score, mean_absolute_error
log(f"- Residual predicted from opponent features alone: R² {r2_score(rva['resid'], pv):.3f}, MAE {mean_absolute_error(rva['resid'], pv):.3f}")
log(f"- Baseline (predict mean residual): MAE {mean_absolute_error(rva['resid'], np.full(len(rva), rtr['resid'].mean())):.3f}")
# permutation importance on val
try:
    imp = permutation_importance(reg, np.nan_to_num(Xva, nan=np.nanmean(Xtr)), rva["resid"].values,
                                 n_repeats=15, random_state=0)
    order = np.argsort(imp.importances_mean)[::-1]
    log("\nTop opponent features by residual-importance:")
    for i in order[:8]:
        log(f"  - `{OPP_FEATS[i]}`: {imp.importances_mean[i]:+.4f} ± {imp.importances_std[i]:.4f}")
except Exception as ex:
    log(f"(permutation importance failed: {ex})")

# --- residual by opponent-skill tier (the headline) ---
log("\n## Residual by opponent behavioral-skill (the headline)")
for col, label in [("o_tr", "nominal trophies"), ("o_beh_gap", "behavioral_gap (plays-like minus is)")]:
    sub = df[df[col].notna()]
    if len(sub) < 40:
        log(f"- {label}: insufficient coverage ({len(sub)})"); continue
    sub = sub.copy(); sub["q"] = pd.qcut(sub[col], 4, labels=["Q1","Q2","Q3","Q4"], duplicates="drop")
    g = sub.groupby("q", observed=True).agg(n=("y","size"), actual_wr=("y","mean"),
                                            mean_maxwp=("max_wp","mean"), mean_resid=("resid","mean"))
    log(f"\nBy {label} quartile:")
    log(g.to_string())

with open(OUT, "w") as f:
    f.write("\n".join(lines))
print(f"\nWROTE {OUT}")
