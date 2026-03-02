#!/bin/bash
# Clash Stats development aliases — source this file: . aliases.sh
# These are shortcuts for common operations during development.

# --- Core paths ---
export CS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CS_DB="${CS_ROOT}/data/clash_royale_history.db"
export CS_EXP_DB="${CS_ROOT}/data/classic_1v1_experiment.db"

# --- Quick DB access ---
alias csdb='sqlite3 "${CS_DB}"'                       # main DB shell
alias csexp='sqlite3 "${CS_EXP_DB}"'                  # experiment DB shell

# --- Docker container shortcuts ---
alias csd='docker exec -it cr-tracker'                 # exec into container
alias cscli='docker exec cr-tracker clash-stats'       # run CLI in container
alias cslogs='docker logs -f cr-tracker'               # tail container logs

# --- Experiment runner ---
alias cs1v1='python -m tracker.experiments.classic_1v1 --main-db "${CS_DB}"'
alias cs1v1api='python -m tracker.experiments.classic_1v1 --main-db "${CS_DB}" --fetch-profiles'
alias cs1v1json='python -m tracker.experiments.classic_1v1 --main-db "${CS_DB}" --json'

# --- Quick queries (functions for parameterization) ---
csq() {
    # Run a SQL query against the main DB
    sqlite3 -header -column "${CS_DB}" "$1"
}

csqe() {
    # Run a SQL query against the experiment DB
    sqlite3 -header -column "${CS_EXP_DB}" "$1"
}

cstag() {
    # Look up a player tag across all battles: cstag '#VJQQYUVLR'
    local tag="$1"
    sqlite3 -header -column "${CS_DB}" "
        SELECT opponent_tag, opponent_name,
               MAX(opponent_starting_trophies) as max_tr,
               COUNT(*) as games,
               game_mode_name
        FROM battles
        WHERE opponent_tag = '${tag}'
        GROUP BY game_mode_name;
    "
}

csopp() {
    # Search opponent by name substring: csopp 'Raze'
    local name="$1"
    sqlite3 -header -column "${CS_DB}" "
        SELECT DISTINCT opponent_tag, opponent_name,
               MAX(opponent_starting_trophies) as max_tr,
               COUNT(*) as appearances
        FROM battles
        WHERE opponent_name LIKE '%${name}%'
        GROUP BY opponent_tag
        ORDER BY max_tr DESC;
    "
}

echo "Clash Stats aliases loaded. CS_DB=${CS_DB}"
