/* Clash Royale Analytics Dashboard — Chart.js + vanilla fetch */

const POLL_INTERVAL = 3 * 60 * 1000; // 3 minutes

// Chart.js global defaults for dark theme
Chart.defaults.color = "#8b8fa3";
Chart.defaults.borderColor = "rgba(42, 45, 58, 0.8)";
Chart.defaults.font.family = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";

let trophyChart = null;
let archetypeChart = null;
let crownChart = null;
let timeChart = null;

// ─── Helpers ────────────────────────────────────────────────────

function wrClass(wr) {
    if (wr >= 55) return "wr-good";
    if (wr >= 45) return "wr-mid";
    return "wr-bad";
}

function formatBattleTime(bt) {
    if (!bt || bt.length < 15) return bt || "";
    // "20260214T180000.000Z" → "Feb 14 18:00"
    const y = bt.slice(0, 4), m = bt.slice(4, 6), d = bt.slice(6, 8);
    const h = bt.slice(9, 11), mn = bt.slice(11, 13);
    const date = new Date(`${y}-${m}-${d}T${h}:${mn}:00Z`);
    return date.toLocaleDateString("en-US", { month: "short", day: "numeric" }) +
        " " + date.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false });
}

function formatShortDate(bt) {
    if (!bt || bt.length < 8) return bt || "";
    const m = bt.slice(4, 6), d = bt.slice(6, 8);
    return `${parseInt(m)}/${parseInt(d)}`;
}

// ─── Data fetching ──────────────────────────────────────────────

async function fetchAll() {
    const [overview, trophyHistory, matchups, recent, streaks] = await Promise.all([
        fetch("/api/overview").then(r => r.json()),
        fetch("/api/trophy-history").then(r => r.json()),
        fetch("/api/matchups").then(r => r.json()),
        fetch("/api/recent").then(r => r.json()),
        fetch("/api/streaks").then(r => r.json()),
    ]);
    renderOverview(overview);
    renderTrophyChart(trophyHistory);
    renderMatchups(matchups);
    renderRecentBattles(recent);
    renderStreaks(streaks);
    document.getElementById("last-updated").textContent = new Date().toLocaleTimeString();
}

// ─── Overview header ────────────────────────────────────────────

function renderOverview(data) {
    const api = data.api_stats || {};
    const tracked = data.tracked || {};
    const diff = data.snapshot_diff;

    document.getElementById("player-name").textContent = api.name || "Clash Royale";
    const clanEl = document.getElementById("clan-name");
    if (api.clan_name) {
        clanEl.textContent = api.clan_name;
        clanEl.style.display = "";
    } else {
        clanEl.style.display = "none";
    }

    document.getElementById("trophies").textContent =
        api.trophies != null ? api.trophies.toLocaleString() : "--";
    document.getElementById("tracked-total").textContent =
        tracked.total != null ? tracked.total.toLocaleString() : "--";

    const total = tracked.total || 0;
    const wins = tracked.wins || 0;
    const wr = total > 0 ? (wins / total * 100).toFixed(1) + "%" : "--";
    document.getElementById("win-rate").textContent = wr;

    const diffPill = document.getElementById("diff-pill");
    if (diff && (diff.trophies || diff.wins || diff.losses)) {
        const parts = [];
        if (diff.trophies) parts.push((diff.trophies > 0 ? "+" : "") + diff.trophies + " tr");
        if (diff.wins) parts.push("+" + diff.wins + "W");
        if (diff.losses) parts.push("+" + diff.losses + "L");
        document.getElementById("diff-summary").textContent = parts.join("  ");
        diffPill.style.display = "";
    } else {
        diffPill.style.display = "none";
    }
}

// ─── Trophy chart ───────────────────────────────────────────────

function renderTrophyChart(history) {
    const ctx = document.getElementById("trophyChart").getContext("2d");

    const labels = history.map(h => formatShortDate(h.battle_time));
    const dataPoints = history.map(h => h.trophies);
    const colors = history.map(h =>
        h.result === "win" ? "#34d399" : h.result === "loss" ? "#f87171" : "#8b8fa3"
    );

    if (trophyChart) trophyChart.destroy();
    trophyChart = new Chart(ctx, {
        type: "line",
        data: {
            labels,
            datasets: [{
                label: "Trophies",
                data: dataPoints,
                borderColor: "#4f8cff",
                borderWidth: 2,
                pointBackgroundColor: colors,
                pointRadius: history.length > 100 ? 1 : 3,
                pointHoverRadius: 5,
                fill: {
                    target: "origin",
                    above: "rgba(79, 140, 255, 0.08)",
                },
                tension: 0.2,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        title: (items) => {
                            const i = items[0].dataIndex;
                            return formatBattleTime(history[i].battle_time);
                        },
                        afterLabel: (item) => {
                            const h = history[item.dataIndex];
                            const change = h.player_trophy_change || 0;
                            return (h.result === "win" ? "Win" : h.result === "loss" ? "Loss" : "Draw") +
                                "  (" + (change >= 0 ? "+" : "") + change + ")";
                        },
                    },
                },
            },
            scales: {
                x: {
                    ticks: { maxTicksLimit: 15 },
                },
                y: {
                    beginAtZero: false,
                },
            },
        },
    });
}

// ─── Matchups ───────────────────────────────────────────────────

function renderMatchups(data) {
    // Archetype chart
    const archetypes = data.archetypes || [];
    const ctx = document.getElementById("archetypeChart").getContext("2d");

    if (archetypeChart) archetypeChart.destroy();
    archetypeChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: archetypes.map(a => a.archetype),
            datasets: [{
                label: "Win Rate %",
                data: archetypes.map(a => a.win_rate),
                backgroundColor: archetypes.map(a =>
                    a.win_rate >= 55 ? "rgba(52, 211, 153, 0.7)" :
                    a.win_rate >= 45 ? "rgba(251, 191, 36, 0.7)" :
                    "rgba(248, 113, 113, 0.7)"
                ),
                borderRadius: 4,
            }, {
                label: "Games",
                data: archetypes.map(a => a.total),
                type: "line",
                borderColor: "#4f8cff",
                pointRadius: 3,
                yAxisID: "y1",
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: "y",
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        afterLabel: (item) => {
                            const a = archetypes[item.dataIndex];
                            return `${a.wins}W-${a.losses}L (${a.total} games)`;
                        },
                    },
                },
            },
            scales: {
                x: { beginAtZero: true, max: 100, title: { display: true, text: "Win Rate %" } },
                y1: { position: "right", display: false },
            },
        },
    });

    // Card matchup tables
    const matchups = data.card_matchups || [];
    const sorted = [...matchups].sort((a, b) => b.win_rate - a.win_rate);
    const best = sorted.slice(0, 8);
    const worst = sorted.slice(-8).reverse();

    fillMatchupTable("best-matchups", best);
    fillMatchupTable("worst-matchups", worst);
}

function fillMatchupTable(id, rows) {
    const tbody = document.querySelector(`#${id} tbody`);
    tbody.innerHTML = rows.map(r =>
        `<tr>
            <td>${r.card_name}</td>
            <td>${r.times_faced}</td>
            <td class="${wrClass(r.win_rate)}">${r.win_rate}%</td>
        </tr>`
    ).join("");
}

// ─── Recent battles ─────────────────────────────────────────────

function renderRecentBattles(battles) {
    const feed = document.getElementById("battle-feed");
    feed.innerHTML = battles.map(b => {
        const icon = b.result === "win" ? "\u2714" : b.result === "loss" ? "\u2718" : "\u2014";
        const resultClass = "result-" + b.result;
        const change = b.player_trophy_change || 0;
        const changeStr = change >= 0 ? "+" + change : "" + change;
        const changeClass = change >= 0 ? "trophy-pos" : "trophy-neg";
        return `<div class="battle-row">
            <span class="result-icon ${resultClass}">${icon}</span>
            <span class="score">${b.player_crowns}-${b.opponent_crowns}</span>
            <span class="opponent">${b.opponent_name || "Unknown"}</span>
            <span class="trophy-change ${changeClass}">${changeStr}</span>
            <span class="battle-time">${formatBattleTime(b.battle_time)}</span>
        </div>`;
    }).join("");
}

// ─── Streaks & rolling ──────────────────────────────────────────

function renderStreaks(data) {
    const streakData = data.streaks || {};
    const cardsEl = document.getElementById("streak-cards");

    const cards = [];
    const cs = streakData.current_streak;
    if (cs) {
        const icon = cs.type === "win" ? "\uD83D\uDD25" : "\u2744\uFE0F";
        const label = cs.length === 1 ? cs.type : cs.type === "win" ? "wins" : "losses";
        cards.push({ label: "Current Streak", value: `${icon} ${cs.length} ${label}`, detail: `${cs.start_trophies} \u2192 ${cs.end_trophies}` });
    }
    const ws = streakData.longest_win_streak;
    if (ws) {
        cards.push({ label: "Best Win Run", value: `${ws.length} wins`, detail: `${ws.start_trophies} \u2192 ${ws.end_trophies}` });
    }
    const ls = streakData.longest_loss_streak;
    if (ls) {
        cards.push({ label: "Worst Tilt", value: `${ls.length} losses`, detail: `${ls.start_trophies} \u2192 ${ls.end_trophies}` });
    }

    cardsEl.innerHTML = cards.map(c =>
        `<div class="streak-card">
            <div class="streak-label">${c.label}</div>
            <div class="streak-value">${c.value}</div>
            <div class="streak-detail">${c.detail}</div>
        </div>`
    ).join("");

    // Rolling stats
    renderRollingCol("rolling-35", data.rolling_35);
    renderRollingCol("rolling-10", data.rolling_10);

    // Crown distribution doughnut
    renderCrownChart(data.crown_distribution);

    // Time of day chart
    renderTimeChart(data.time_of_day);
}

function renderRollingCol(id, stats) {
    if (!stats || stats.total === 0) {
        document.getElementById(id).innerHTML = "<em>No data</em>";
        return;
    }
    const rows = [
        ["Win Rate", `${stats.win_rate}%`],
        ["Record", `${stats.wins}W-${stats.losses}L${stats.draws ? "-" + stats.draws + "D" : ""}`],
        ["3-Crowns", `${stats.three_crowns}`],
        ["Avg Crowns", `${stats.avg_crowns}`],
        ["Trophy \u0394", `${stats.trophy_change >= 0 ? "+" : ""}${stats.trophy_change}`],
    ];
    document.getElementById(id).innerHTML = rows.map(([l, v]) =>
        `<div class="rolling-stat-row"><span class="rs-label">${l}</span><span class="rs-value">${v}</span></div>`
    ).join("");
}

function renderCrownChart(dist) {
    if (!dist) return;
    const ctx = document.getElementById("crownChart").getContext("2d");

    const winCrowns = [dist.win?.["1"] || 0, dist.win?.["2"] || 0, dist.win?.["3"] || 0];
    const lossCrowns = [dist.loss?.["0"] || 0, dist.loss?.["1"] || 0, dist.loss?.["2"] || 0];

    if (crownChart) crownChart.destroy();
    crownChart = new Chart(ctx, {
        type: "doughnut",
        data: {
            labels: ["1-crown W", "2-crown W", "3-crown W", "0-crown L", "1-crown L", "2-crown L"],
            datasets: [{
                data: [...winCrowns, ...lossCrowns],
                backgroundColor: [
                    "rgba(52, 211, 153, 0.5)",
                    "rgba(52, 211, 153, 0.7)",
                    "rgba(52, 211, 153, 0.9)",
                    "rgba(248, 113, 113, 0.5)",
                    "rgba(248, 113, 113, 0.7)",
                    "rgba(248, 113, 113, 0.9)",
                ],
                borderWidth: 0,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: "right", labels: { boxWidth: 12, padding: 8, font: { size: 11 } } },
            },
        },
    });
}

function renderTimeChart(timeData) {
    if (!timeData || timeData.length === 0) return;
    const ctx = document.getElementById("timeChart").getContext("2d");

    if (timeChart) timeChart.destroy();
    timeChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: timeData.map(t => `${String(t.hour).padStart(2, "0")}:00`),
            datasets: [{
                label: "Win Rate %",
                data: timeData.map(t => t.win_rate),
                backgroundColor: timeData.map(t =>
                    t.win_rate >= 55 ? "rgba(52, 211, 153, 0.7)" :
                    t.win_rate >= 45 ? "rgba(251, 191, 36, 0.7)" :
                    "rgba(248, 113, 113, 0.7)"
                ),
                borderRadius: 4,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        afterLabel: (item) => {
                            const t = timeData[item.dataIndex];
                            return `${t.wins}W / ${t.total} games`;
                        },
                    },
                },
            },
            scales: {
                y: { beginAtZero: true, max: 100, title: { display: true, text: "WR %" } },
            },
        },
    });
}

// ─── Monte Carlo simulation ──────────────────────────────────────

async function fetchSimulation() {
    try {
        const resp = await fetch("/api/simulation");
        if (!resp.ok) return;
        const data = await resp.json();
        renderSimulation(data);
    } catch (e) {
        // No simulation data yet — hide section
    }
}

function renderSimulation(data) {
    const section = document.getElementById("sim-section");
    section.style.display = "";

    // Timestamp
    if (data.computed_at) {
        const ts = new Date(data.computed_at);
        document.getElementById("sim-timestamp").textContent =
            ts.toLocaleString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
    }

    // Threat ranking — use personal if available, fall back to corpus
    const threats = data.personal_threats || data.corpus_threats || [];
    const tbody = document.querySelector("#threat-table tbody");
    tbody.innerHTML = threats.slice(0, 20).map(t => {
        const pct = (t.posterior_mean * 100).toFixed(1);
        const ci = `[${(t.ci_low * 100).toFixed(0)}, ${(t.ci_high * 100).toFixed(0)}]`;
        return `<tr>
            <td>${t.archetype}</td>
            <td>${t.wins}</td>
            <td>${t.losses}</td>
            <td class="${wrClass(t.posterior_mean * 100)}">${pct}%</td>
            <td class="ci-col">${ci}</td>
        </tr>`;
    }).join("");

    // Card threat matrix — worst cards (lowest win rate)
    const interactions = data.personal_card_interactions || data.card_interactions || {};
    const cardList = Object.entries(interactions)
        .map(([name, d]) => ({ name, ...d }))
        .filter(c => c.total >= 10)
        .sort((a, b) => a.win_rate - b.win_rate);

    const worstCards = cardList.slice(0, 15);
    const bestCards = cardList.slice(-10).reverse();
    const cardRows = [...worstCards, { name: "───", total: "", win_rate: null, ci_low: null, ci_high: null }, ...bestCards];

    const cardTbody = document.querySelector("#card-threat-table tbody");
    cardTbody.innerHTML = cardRows.map(c => {
        if (c.win_rate === null) {
            return `<tr class="separator"><td colspan="4">─── Best ───</td></tr>`;
        }
        const pct = (c.win_rate * 100).toFixed(1);
        const ci = `[${(c.ci_low * 100).toFixed(0)}, ${(c.ci_high * 100).toFixed(0)}]`;
        return `<tr>
            <td>${c.name}</td>
            <td>${c.total}</td>
            <td class="${wrClass(c.win_rate * 100)}">${pct}%</td>
            <td class="ci-col">${ci}</td>
        </tr>`;
    }).join("");

    // Sub-archetypes
    const subSection = document.getElementById("sub-archetype-section");
    const subs = data.sub_archetypes || {};
    const subEntries = Object.entries(subs).filter(([, v]) => v.length > 0);
    if (subEntries.length > 0) {
        subSection.innerHTML = `<h3>Sub-Archetype Breakdown</h3>` +
            subEntries.map(([wc, clusters]) => {
                const rows = clusters.map(c => {
                    const sig = c.signature_cards.slice(0, 5).join(", ");
                    const wr = (c.win_rate * 100).toFixed(1);
                    return `<tr>
                        <td>${sig}</td>
                        <td>${c.count}</td>
                        <td>${c.variants}</td>
                        <td class="${wrClass(c.win_rate * 100)}">${wr}%</td>
                        <td>${c.avg_elixir}</td>
                    </tr>`;
                }).join("");
                return `<details open><summary>${wc} (${clusters.length} sub-types)</summary>
                    <table class="matchup-table sub-table">
                        <thead><tr><th>Signature Cards</th><th>Games</th><th>Variants</th><th>WR</th><th>Avg Elixir</th></tr></thead>
                        <tbody>${rows}</tbody>
                    </table></details>`;
            }).join("");
    }
}

// ─── Embeddings 3D scatter plot (Plotly) + Timeline slider ───────

let embeddingData = null;  // store for click lookups
let embeddingDates = [];   // sorted unique personal dates for slider
let embeddingPlotInit = false;

async function fetchEmbeddings() {
    try {
        const resp = await fetch("/api/embeddings");
        if (!resp.ok) return;
        const data = await resp.json();
        embeddingData = data.points;
        renderEmbeddingChart(data.points);
        initTimelineSlider(data.points);
    } catch (e) {
        // No embeddings yet — hide section
    }
}

// Parse "20260228T035121.000Z" to Date
function parseBattleTime(bt) {
    if (!bt || bt.length < 15) return null;
    const y = bt.slice(0, 4), m = bt.slice(4, 6), d = bt.slice(6, 8);
    const h = bt.slice(9, 11), mn = bt.slice(11, 13), s = bt.slice(13, 15);
    return new Date(`${y}-${m}-${d}T${h}:${mn}:${s}Z`);
}

function formatSliderDate(date) {
    return date.toLocaleDateString("en-US", { month: "short", day: "numeric" }) +
        " " + date.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false });
}

function initTimelineSlider(points) {
    // Collect all dates from personal games
    const personalPoints = points.filter(p => p.corpus === "personal" && p.battle_time);
    if (personalPoints.length < 2) return;  // no slider needed for <2 games

    const dates = personalPoints
        .map(p => parseBattleTime(p.battle_time))
        .filter(d => d)
        .sort((a, b) => a - b);
    if (dates.length < 2) return;

    embeddingDates = dates;

    const control = document.getElementById("timeline-control");
    control.style.display = "";

    const slider = document.getElementById("timeline-slider");
    slider.min = 0;
    slider.max = dates.length - 1;
    slider.value = dates.length - 1;

    document.getElementById("timeline-min-label").textContent = formatSliderDate(dates[0]);
    document.getElementById("timeline-max-label").textContent = formatSliderDate(dates[dates.length - 1]);
    document.getElementById("timeline-date").textContent = `All ${dates.length} personal games`;

    slider.addEventListener("input", onTimelineSlide);
    document.getElementById("timeline-reset").addEventListener("click", () => {
        slider.value = dates.length - 1;
        onTimelineSlide();
    });
}

function onTimelineSlide() {
    const slider = document.getElementById("timeline-slider");
    const idx = parseInt(slider.value);
    const cutoffDate = embeddingDates[idx];

    const showing = idx + 1;
    const total = embeddingDates.length;
    const label = showing === total
        ? `All ${total} personal games`
        : `${showing} of ${total} games — through ${formatSliderDate(cutoffDate)}`;
    document.getElementById("timeline-date").textContent = label;

    // Filter personal points by date
    updateEmbeddingVisibility(cutoffDate);
}

function updateEmbeddingVisibility(cutoffDate) {
    if (!embeddingData) return;

    const personal = embeddingData.filter(p => p.corpus === "personal");
    const visible = personal.filter(p => {
        const d = parseBattleTime(p.battle_time);
        return d && d <= cutoffDate;
    });

    const personalWins = visible.filter(p => p.result === "win");
    const personalLosses = visible.filter(p => p.result === "loss");

    // Update traces 2 and 3 (personal wins and losses) via Plotly.restyle
    Plotly.restyle("embeddingPlot", {
        x: [personalWins.map(p => p.x)],
        y: [personalWins.map(p => p.y)],
        z: [personalWins.map(p => p.z)],
        text: [personalWins.map(p => `${p.opponent || "?"} — ${p.result} (${formatSliderDate(parseBattleTime(p.battle_time))})`)],
        customdata: [personalWins.map(p => p.battle_id)],
    }, [2]);

    Plotly.restyle("embeddingPlot", {
        x: [personalLosses.map(p => p.x)],
        y: [personalLosses.map(p => p.y)],
        z: [personalLosses.map(p => p.z)],
        text: [personalLosses.map(p => `${p.opponent || "?"} — ${p.result} (${formatSliderDate(parseBattleTime(p.battle_time))})`)],
        customdata: [personalLosses.map(p => p.battle_id)],
    }, [3]);
}

function renderEmbeddingChart(points) {
    if (!points || points.length === 0) return;

    const section = document.getElementById("embeddings-section");
    section.style.display = "";

    // Separate into 4 categories matching previous color scheme
    const corpusWins = points.filter(p => p.result === "win" && p.corpus !== "personal");
    const corpusLosses = points.filter(p => p.result === "loss" && p.corpus !== "personal");
    const personalWins = points.filter(p => p.result === "win" && p.corpus === "personal");
    const personalLosses = points.filter(p => p.result === "loss" && p.corpus === "personal");

    const makeTrace = (arr, name, color, size, opacity, border) => ({
        type: "scatter3d",
        mode: "markers",
        name: name,
        x: arr.map(p => p.x),
        y: arr.map(p => p.y),
        z: arr.map(p => p.z),
        text: arr.map(p => {
            const dateStr = p.battle_time ? formatSliderDate(parseBattleTime(p.battle_time)) : "";
            return `${p.opponent || "?"} — ${p.result}${dateStr ? " (" + dateStr + ")" : ""}`;
        }),
        customdata: arr.map(p => p.battle_id),
        hovertemplate: "%{text}<extra></extra>",
        marker: {
            size: size,
            color: color,
            opacity: opacity,
            line: border ? { color: "#ffffff", width: 0.5 } : undefined,
        },
    });

    const data = [
        makeTrace(corpusWins,     "Corpus Wins",    "rgb(52, 211, 153)",  2.5, 0.25, false),
        makeTrace(corpusLosses,   "Corpus Losses",  "rgb(248, 113, 113)", 2.5, 0.25, false),
        makeTrace(personalWins,   "My Wins",        "rgb(52, 211, 153)",  5,   0.9,  true),
        makeTrace(personalLosses, "My Losses",      "rgb(248, 113, 113)", 5,   0.9,  true),
    ];

    const axisStyle = {
        gridcolor: "#2a2d3a",
        zerolinecolor: "#2a2d3a",
        color: "#8b8fa3",
        backgroundcolor: "#1a1d27",
    };

    const layout = {
        paper_bgcolor: "#1a1d27",
        font: { color: "#8b8fa3", family: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" },
        margin: { l: 0, r: 0, t: 0, b: 0 },
        scene: {
            xaxis: { ...axisStyle, title: "UMAP-1" },
            yaxis: { ...axisStyle, title: "UMAP-2" },
            zaxis: { ...axisStyle, title: "UMAP-3" },
            bgcolor: "#1a1d27",
        },
        legend: {
            font: { size: 11, color: "#8b8fa3" },
            bgcolor: "rgba(26, 29, 39, 0.8)",
        },
        showlegend: true,
    };

    const config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ["toImage", "sendDataToCloud"],
        displaylogo: false,
    };

    Plotly.newPlot("embeddingPlot", data, layout, config);
    embeddingPlotInit = true;

    // Click handler — customdata carries battle_id directly
    document.getElementById("embeddingPlot").on("plotly_click", function(eventData) {
        if (eventData.points.length > 0) {
            const battleId = eventData.points[0].customdata;
            if (battleId) {
                fetchSimilar(battleId);
            }
        }
    });
}

async function fetchSimilar(battleId) {
    try {
        const resp = await fetch(`/api/similar/${battleId}`);
        if (!resp.ok) return;
        const data = await resp.json();
        renderSimilarPanel(battleId, data);
    } catch (e) {
        console.error("Failed to fetch similar games:", e);
    }
}

function similarRows(games) {
    return games.map(s => {
        const score = `${s.player_crowns ?? "?"}-${s.opponent_crowns ?? "?"}`;
        const resultClass = s.result === "win" ? "wr-good" : s.result === "loss" ? "wr-bad" : "";
        const deck = (s.opponent_deck || []).join(", ");
        const pct = (s.percentile * 100).toFixed(1);
        const pctClass = s.percentile >= 0.95 ? "wr-good" : s.percentile >= 0.80 ? "wr-mid" : "wr-bad";
        const kernel = s.similarity.toFixed(3);
        return `<tr>
            <td>${s.opponent_name || "Unknown"}</td>
            <td class="${resultClass}">${s.result || "?"}</td>
            <td>${score}</td>
            <td class="${pctClass}">Top ${pct}%</td>
            <td class="kernel-col">${kernel}</td>
            <td class="archetype-col">${s.archetype || "?"}</td>
            <td class="deck-col" title="${deck}">${deck}</td>
        </tr>`;
    }).join("");
}

function renderSimilarPanel(battleId, data) {
    const panel = document.getElementById("similar-panel");
    panel.style.display = "";
    document.getElementById("similar-ref").textContent = battleId.slice(0, 12) + "…";

    const thead = `<thead><tr><th>Opponent</th><th>Result</th><th>Score</th><th>Rank</th><th>Kernel</th><th>Archetype</th><th>Deck</th></tr></thead>`;
    let html = "";

    if (data.personal && data.personal.length > 0) {
        html += `<h3 class="similar-subhead">My Similar Games</h3>
            <table class="matchup-table">${thead}<tbody>${similarRows(data.personal)}</tbody></table>`;
    }

    if (data.corpus && data.corpus.length > 0) {
        html += `<h3 class="similar-subhead">Corpus Similar Games</h3>
            <table class="matchup-table">${thead}<tbody>${similarRows(data.corpus)}</tbody></table>`;
    }

    document.getElementById("similar-tables").innerHTML = html;
}

// ─── Tilt detection ─────────────────────────────────────────────

async function fetchTilt() {
    try {
        const resp = await fetch("/api/tilt");
        if (!resp.ok) return;
        const data = await resp.json();
        renderTiltBanner(data);
    } catch (e) {
        // Tilt detection not available
    }
}

function renderTiltBanner(data) {
    const banner = document.getElementById("tilt-banner");

    if (data.level === "none") {
        banner.style.display = "none";
        return;
    }

    banner.style.display = "flex";
    banner.className = "tilt-banner tilt-" + data.level;

    const icons = { warning: "\u26a0\ufe0f", tilting: "\ud83d\udd34", severe: "\ud83d\uded1" };
    const labels = { warning: "TILT WARNING", tilting: "TILTING", severe: "TILT \u2014 STOP PLAYING" };

    document.getElementById("tilt-icon").textContent = icons[data.level] || "";
    document.getElementById("tilt-label").textContent = labels[data.level] || data.level.toUpperCase();
    document.getElementById("tilt-message").textContent = data.message;

    const stats = [`${data.recent_record}`,
        `Streak: ${data.consecutive_losses}L`,
        `Leak: ${data.avg_leak} avg / ${data.max_leak} max`];
    if (data.embedding_matches > 0) {
        stats.push(`TCN tilt matches: ${data.embedding_matches}`);
    }
    document.getElementById("tilt-stats").textContent = stats.join("  \u00b7  ");
}

// ─── Init & poll ────────────────────────────────────────────────

fetchAll();
fetchTilt();
fetchSimulation();
fetchEmbeddings();
setInterval(fetchAll, POLL_INTERVAL);
setInterval(fetchTilt, POLL_INTERVAL);
setInterval(fetchSimulation, POLL_INTERVAL);
