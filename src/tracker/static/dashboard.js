/* Clash Royale Analytics Dashboard — BigPipe + paginated data loading */

const POLL_INTERVAL = 3 * 60 * 1000; // 3 minutes
const PAGE_SIZE = 1000;

// Chart.js global defaults for dark theme
Chart.defaults.color = "#8b8fa3";
Chart.defaults.borderColor = "rgba(42, 45, 58, 0.8)";
Chart.defaults.font.family = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";

let trophyChart = null;
let archetypeChart = null;
let crownChart = null;
let timeChart = null;
let wpChart = null;

// Accumulated data for paginated endpoints
let trophyHistory = [];
let trophyTotal = 0;

// ─── Helpers ────────────────────────────────────────────────────

function wrClass(wr) {
    if (wr >= 55) return "wr-good";
    if (wr >= 45) return "wr-mid";
    return "wr-bad";
}

function formatBattleTime(bt) {
    if (!bt || bt.length < 15) return bt || "";
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

// ─── Section loading state ──────────────────────────────────────

function sectionLoading(sectionId) {
    const el = document.getElementById(sectionId);
    if (el) {
        let spinner = el.querySelector(".section-spinner");
        if (!spinner) {
            spinner = document.createElement("div");
            spinner.className = "section-spinner";
            spinner.textContent = "Loading...";
            el.appendChild(spinner);
        }
        spinner.style.display = "";
    }
}

function sectionReady(sectionId) {
    const el = document.getElementById(sectionId);
    if (el) {
        const spinner = el.querySelector(".section-spinner");
        if (spinner) spinner.style.display = "none";
    }
}

function sectionProgress(sectionId, loaded, total) {
    const el = document.getElementById(sectionId);
    if (!el) return;
    let spinner = el.querySelector(".section-spinner");
    if (!spinner) {
        spinner = document.createElement("div");
        spinner.className = "section-spinner";
        el.appendChild(spinner);
    }
    spinner.style.display = "";
    spinner.textContent = `Loading ${loaded.toLocaleString()} / ${total.toLocaleString()}...`;
}

// ─── BigPipe: independent section fetchers ──────────────────────

async function fetchOverview() {
    try {
        const data = await fetch("/api/overview").then(r => r.json());
        renderOverview(data);
    } catch (e) {
        console.error("Overview fetch failed:", e);
    }
}

async function fetchTrophyHistory() {
    try {
        sectionLoading("trophy-section");
        trophyHistory = [];
        trophyTotal = 0;
        let page = 0;
        let hasMore = true;

        while (hasMore) {
            const resp = await fetch(`/api/trophy-history?page=${page}&per_page=${PAGE_SIZE}`);
            const data = await resp.json();
            trophyHistory.push(...data.data);
            trophyTotal = data.total;
            hasMore = data.has_more;
            page++;

            // Render progressively — first page renders the chart, subsequent pages extend it
            if (page === 1) {
                renderTrophyChart(trophyHistory);
            } else {
                updateTrophyChart(trophyHistory);
            }
            if (hasMore) {
                sectionProgress("trophy-section", trophyHistory.length, trophyTotal);
            }
        }
        sectionReady("trophy-section");
    } catch (e) {
        console.error("Trophy history fetch failed:", e);
        sectionReady("trophy-section");
    }
}

async function fetchMatchups() {
    try {
        sectionLoading("matchups-section");
        const data = await fetch("/api/matchups").then(r => r.json());
        renderMatchups(data);
        sectionReady("matchups-section");
    } catch (e) {
        console.error("Matchups fetch failed:", e);
        sectionReady("matchups-section");
    }
}

async function fetchRecent() {
    try {
        sectionLoading("recent-section");
        const data = await fetch("/api/recent").then(r => r.json());
        renderRecentBattles(data);
        sectionReady("recent-section");
    } catch (e) {
        console.error("Recent fetch failed:", e);
        sectionReady("recent-section");
    }
}

async function fetchStreaks() {
    try {
        sectionLoading("streaks-section");
        const data = await fetch("/api/streaks").then(r => r.json());
        renderStreaks(data);
        sectionReady("streaks-section");
    } catch (e) {
        console.error("Streaks fetch failed:", e);
        sectionReady("streaks-section");
    }
}

// Fire all sections independently — each renders as soon as its data arrives
function fetchAll() {
    fetchOverview();
    fetchTrophyHistory();
    fetchMatchups();
    fetchRecent();
    fetchStreaks();
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

function buildTrophyChartConfig(history) {
    return {
        type: "line",
        data: {
            labels: history.map(h => formatShortDate(h.battle_time)),
            datasets: [{
                label: "Trophies",
                data: history.map(h => h.trophies),
                borderColor: "#4f8cff",
                borderWidth: 2,
                pointBackgroundColor: history.map(h =>
                    h.result === "win" ? "#34d399" : h.result === "loss" ? "#f87171" : "#8b8fa3"
                ),
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
            animation: false,
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
                x: { ticks: { maxTicksLimit: 15 } },
                y: { beginAtZero: false },
            },
        },
    };
}

function renderTrophyChart(history) {
    const ctx = document.getElementById("trophyChart").getContext("2d");
    if (trophyChart) trophyChart.destroy();
    trophyChart = new Chart(ctx, buildTrophyChartConfig(history));
}

function updateTrophyChart(history) {
    if (!trophyChart) return renderTrophyChart(history);
    const ds = trophyChart.data.datasets[0];
    ds.data = history.map(h => h.trophies);
    ds.pointBackgroundColor = history.map(h =>
        h.result === "win" ? "#34d399" : h.result === "loss" ? "#f87171" : "#8b8fa3"
    );
    ds.pointRadius = history.length > 100 ? 1 : 3;
    trophyChart.data.labels = history.map(h => formatShortDate(h.battle_time));
    trophyChart.update("none"); // no animation for incremental updates
}

// ─── Matchups ───────────────────────────────────────────────────

function renderMatchups(data) {
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
        const bid = b.battle_id || "";
        return `<div class="battle-row clickable" onclick="showWPCurve('${bid}')" title="Click for WP curve">
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

    renderRollingCol("rolling-35", data.rolling_35);
    renderRollingCol("rolling-10", data.rolling_10);
    renderCrownChart(data.crown_distribution);
    renderTimeChart(data.time_of_day, data.corpus_traffic);
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

function renderTimeChart(timeData, corpusTraffic) {
    if (!timeData || timeData.length === 0) return;
    const ctx = document.getElementById("timeChart").getContext("2d");

    const trafficByHour = {};
    if (corpusTraffic) {
        for (const t of corpusTraffic) trafficByHour[t.hour] = t.traffic_index;
    }

    const datasets = [{
        label: "Win Rate %",
        data: timeData.map(t => t.win_rate),
        backgroundColor: timeData.map(t =>
            t.win_rate >= 55 ? "rgba(52, 211, 153, 0.7)" :
            t.win_rate >= 45 ? "rgba(251, 191, 36, 0.7)" :
            "rgba(248, 113, 113, 0.7)"
        ),
        borderRadius: 4,
        yAxisID: "y",
    }];

    if (corpusTraffic && corpusTraffic.length > 0) {
        datasets.push({
            label: "Global Traffic",
            data: timeData.map(t => trafficByHour[t.hour] ?? null),
            type: "line",
            borderColor: "rgba(79, 140, 255, 0.6)",
            backgroundColor: "rgba(79, 140, 255, 0.1)",
            borderWidth: 2,
            pointRadius: 0,
            fill: true,
            tension: 0.4,
            yAxisID: "y1",
        });
    }

    if (timeChart) timeChart.destroy();
    timeChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: timeData.map(t => `${String(t.hour).padStart(2, "0")}:00`),
            datasets,
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: corpusTraffic && corpusTraffic.length > 0,
                    labels: { boxWidth: 12, padding: 8, font: { size: 11 } },
                },
                tooltip: {
                    callbacks: {
                        afterLabel: (item) => {
                            if (item.datasetIndex === 0) {
                                const t = timeData[item.dataIndex];
                                return `${t.wins}W / ${t.total} games`;
                            }
                            return "Relative corpus activity";
                        },
                    },
                },
            },
            scales: {
                y: { beginAtZero: true, max: 100, title: { display: true, text: "WR %" } },
                y1: {
                    position: "right",
                    beginAtZero: true,
                    max: 100,
                    display: false,
                },
            },
        },
    });
}

// ─── Monte Carlo simulation ──────────────────────────────────────

async function fetchSimulation() {
    try {
        sectionLoading("sim-section");
        const resp = await fetch("/api/simulation");
        if (!resp.ok) { sectionReady("sim-section"); return; }
        const data = await resp.json();
        renderSimulation(data);
        sectionReady("sim-section");
    } catch (e) {
        sectionReady("sim-section");
    }
}

function renderSimulation(data) {
    const section = document.getElementById("sim-section");
    section.style.display = "";

    if (data.computed_at) {
        const ts = new Date(data.computed_at);
        document.getElementById("sim-timestamp").textContent =
            ts.toLocaleString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
    }

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

    const interactions = data.personal_card_interactions || data.card_interactions || {};
    const cardList = Object.entries(interactions)
        .map(([name, d]) => ({ name, ...d }))
        .filter(c => c.total >= 10)
        .sort((a, b) => a.win_rate - b.win_rate);

    const worstCards = cardList.slice(0, 15);
    const bestCards = cardList.slice(-10).reverse();
    const cardRows = [...worstCards, { name: "\u2500\u2500\u2500", total: "", win_rate: null, ci_low: null, ci_high: null }, ...bestCards];

    const cardTbody = document.querySelector("#card-threat-table tbody");
    cardTbody.innerHTML = cardRows.map(c => {
        if (c.win_rate === null) {
            return `<tr class="separator"><td colspan="4">\u2500\u2500\u2500 Best \u2500\u2500\u2500</td></tr>`;
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
                return `<details><summary>${wc} (${clusters.length} sub-types)</summary>
                    <table class="matchup-table sub-table">
                        <thead><tr><th>Signature Cards</th><th>Games</th><th>Variants</th><th>WR</th><th>Avg Elixir</th></tr></thead>
                        <tbody>${rows}</tbody>
                    </table></details>`;
            }).join("");
    }
}

// ─── Embeddings 3D scatter plot (Plotly) + Timeline slider ───────

let embeddingData = null;
let embeddingDates = [];
let embeddingPlotInit = false;

async function fetchEmbeddings() {
    try {
        sectionLoading("embeddings-section");
        const section = document.getElementById("embeddings-section");
        section.style.display = "";

        const resp = await fetch("/api/embeddings");
        if (!resp.ok) { sectionReady("embeddings-section"); return; }
        const data = await resp.json();
        embeddingData = data.points || [];

        renderEmbeddingChart(embeddingData);
        initTimelineSlider(embeddingData);
        sectionReady("embeddings-section");
    } catch (e) {
        sectionReady("embeddings-section");
    }
}

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
    const personalPoints = points.filter(p => p.corpus === "personal" && p.battle_time);
    if (personalPoints.length < 2) return;

    const dates = personalPoints
        .map(p => parseBattleTime(p.battle_time))
        .filter(d => d)
        .sort((a, b) => a - b);
    if (dates.length < 2) return;

    embeddingDates = dates;

    const control = document.getElementById("timeline-control");
    control.style.display = "";

    const sliderLo = document.getElementById("timeline-slider-lo");
    const sliderHi = document.getElementById("timeline-slider-hi");
    sliderLo.min = 0;
    sliderLo.max = dates.length - 1;
    sliderLo.value = 0;
    sliderHi.min = 0;
    sliderHi.max = dates.length - 1;
    sliderHi.value = dates.length - 1;

    document.getElementById("timeline-min-label").textContent = formatSliderDate(dates[0]);
    document.getElementById("timeline-max-label").textContent = formatSliderDate(dates[dates.length - 1]);
    document.getElementById("timeline-date").textContent = `All ${dates.length} personal games`;

    sliderLo.addEventListener("input", onTimelineSlide);
    sliderHi.addEventListener("input", onTimelineSlide);
    document.getElementById("timeline-reset").addEventListener("click", () => {
        sliderLo.value = 0;
        sliderHi.value = dates.length - 1;
        onTimelineSlide();
    });
}

function onTimelineSlide() {
    const sliderLo = document.getElementById("timeline-slider-lo");
    const sliderHi = document.getElementById("timeline-slider-hi");

    // Prevent thumbs from crossing
    let lo = parseInt(sliderLo.value);
    let hi = parseInt(sliderHi.value);
    if (lo > hi) {
        lo = hi;
        sliderLo.value = lo;
    }

    const loDate = embeddingDates[lo];
    const hiDate = embeddingDates[hi];
    const showing = hi - lo + 1;
    const total = embeddingDates.length;

    const label = (lo === 0 && hi === total - 1)
        ? `All ${total} personal games`
        : `${showing} of ${total} games \u2014 ${formatSliderDate(loDate)} to ${formatSliderDate(hiDate)}`;
    document.getElementById("timeline-date").textContent = label;

    updateEmbeddingVisibility(loDate, hiDate);
}

function updateEmbeddingVisibility(loDate, hiDate) {
    if (!embeddingData) return;

    const personal = embeddingData.filter(p => p.corpus === "personal");
    const visible = personal.filter(p => {
        const d = parseBattleTime(p.battle_time);
        return d && d >= loDate && d <= hiDate;
    });

    const personalWins = visible.filter(p => p.result === "win");
    const personalLosses = visible.filter(p => p.result === "loss");

    Plotly.restyle("embeddingPlot", {
        x: [personalWins.map(p => p.x)],
        y: [personalWins.map(p => p.y)],
        z: [personalWins.map(p => p.z)],
        text: [personalWins.map(p => `${p.opponent || "?"} \u2014 ${p.result} (${formatSliderDate(parseBattleTime(p.battle_time))})`)],
        customdata: [personalWins.map(p => p.battle_id)],
    }, [2]);

    Plotly.restyle("embeddingPlot", {
        x: [personalLosses.map(p => p.x)],
        y: [personalLosses.map(p => p.y)],
        z: [personalLosses.map(p => p.z)],
        text: [personalLosses.map(p => `${p.opponent || "?"} \u2014 ${p.result} (${formatSliderDate(parseBattleTime(p.battle_time))})`)],
        customdata: [personalLosses.map(p => p.battle_id)],
    }, [3]);
}

const EMBEDDING_AXIS_STYLE = {
    gridcolor: "#2a2d3a",
    zerolinecolor: "#2a2d3a",
    color: "#8b8fa3",
    backgroundcolor: "#1a1d27",
};

const EMBEDDING_LAYOUT = {
    paper_bgcolor: "#1a1d27",
    font: { color: "#8b8fa3", family: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" },
    margin: { l: 0, r: 0, t: 0, b: 0 },
    scene: {
        xaxis: { ...EMBEDDING_AXIS_STYLE, title: "UMAP-1" },
        yaxis: { ...EMBEDDING_AXIS_STYLE, title: "UMAP-2" },
        zaxis: { ...EMBEDDING_AXIS_STYLE, title: "UMAP-3" },
        bgcolor: "#1a1d27",
    },
    legend: {
        font: { size: 11, color: "#8b8fa3" },
        bgcolor: "rgba(26, 29, 39, 0.8)",
    },
    showlegend: true,
};

const EMBEDDING_CONFIG = {
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ["toImage", "sendDataToCloud"],
    displaylogo: false,
};

function makeEmbeddingTrace(arr, name, color, size, opacity, border) {
    return {
        type: "scatter3d",
        mode: "markers",
        name: name,
        x: arr.map(p => p.x),
        y: arr.map(p => p.y),
        z: arr.map(p => p.z),
        text: arr.map(p => {
            const dateStr = p.battle_time ? formatSliderDate(parseBattleTime(p.battle_time)) : "";
            return `${p.opponent || "?"} \u2014 ${p.result}${dateStr ? " (" + dateStr + ")" : ""}`;
        }),
        customdata: arr.map(p => p.battle_id),
        hovertemplate: "%{text}<extra></extra>",
        marker: {
            size: size,
            color: color,
            opacity: opacity,
            line: border ? { color: "#ffffff", width: 0.5 } : undefined,
        },
    };
}

function renderEmbeddingChart(points) {
    if (!points || points.length === 0) return;

    const section = document.getElementById("embeddings-section");
    section.style.display = "";

    const corpusWins = points.filter(p => p.result === "win" && p.corpus !== "personal");
    const corpusLosses = points.filter(p => p.result === "loss" && p.corpus !== "personal");
    const personalWins = points.filter(p => p.result === "win" && p.corpus === "personal");
    const personalLosses = points.filter(p => p.result === "loss" && p.corpus === "personal");

    const data = [
        makeEmbeddingTrace(corpusWins,     "Corpus Wins",    "rgb(52, 211, 153)",  2.5, 0.25, false),
        makeEmbeddingTrace(corpusLosses,   "Corpus Losses",  "rgb(248, 113, 113)", 2.5, 0.25, false),
        makeEmbeddingTrace(personalWins,   "My Wins",        "rgb(52, 211, 153)",  2.5, 0.9,  true),
        makeEmbeddingTrace(personalLosses, "My Losses",      "rgb(248, 113, 113)", 2.5, 0.9,  true),
    ];

    Plotly.newPlot("embeddingPlot", data, EMBEDDING_LAYOUT, EMBEDDING_CONFIG);
    embeddingPlotInit = true;

    document.getElementById("embeddingPlot").on("plotly_click", function(eventData) {
        if (eventData.points.length > 0) {
            const battleId = eventData.points[0].customdata;
            if (battleId) fetchSimilar(battleId);
        }
    });
}

function extendEmbeddingChart(newPoints) {
    if (!embeddingPlotInit || !newPoints || newPoints.length === 0) return;

    // Bucket new points into the 4 existing traces
    const buckets = [[], [], [], []]; // corpusWin, corpusLoss, personalWin, personalLoss
    for (const p of newPoints) {
        const isPersonal = p.corpus === "personal";
        const isWin = p.result === "win";
        if (!isPersonal && isWin)       buckets[0].push(p);
        else if (!isPersonal && !isWin) buckets[1].push(p);
        else if (isPersonal && isWin)   buckets[2].push(p);
        else                            buckets[3].push(p);
    }

    for (let i = 0; i < 4; i++) {
        if (buckets[i].length === 0) continue;
        Plotly.extendTraces("embeddingPlot", {
            x: [buckets[i].map(p => p.x)],
            y: [buckets[i].map(p => p.y)],
            z: [buckets[i].map(p => p.z)],
            text: [buckets[i].map(p => {
                const dateStr = p.battle_time ? formatSliderDate(parseBattleTime(p.battle_time)) : "";
                return `${p.opponent || "?"} \u2014 ${p.result}${dateStr ? " (" + dateStr + ")" : ""}`;
            })],
            customdata: [buckets[i].map(p => p.battle_id)],
        }, [i]);
    }
}

// ─── Stereogram rendering ───────────────────────────────────────

let stereoActive = false;
let stereoTraces = null;  // cached traces for stereo rendering
let lastMonoCamera = null;
let stereoPollId = null;  // requestAnimationFrame ID for camera sync

function getStereoSeparation() {
    return parseFloat(document.getElementById("stereo-separation").value) || 0.07;
}

function getStereoMethod() {
    return document.getElementById("stereo-method").value;  // "cross" or "parallel"
}

function offsetCamera(camera, eyeOffset) {
    // Rotate the camera eye position around the center by eyeOffset radians
    // in the horizontal plane (around the up vector)
    const eye = camera.eye || { x: 1.25, y: 1.25, z: 1.25 };
    const center = camera.center || { x: 0, y: 0, z: 0 };
    const up = camera.up || { x: 0, y: 0, z: 1 };

    // Vector from center to eye
    const dx = eye.x - center.x;
    const dy = eye.y - center.y;
    const dz = eye.z - center.z;

    // Rotate around the up vector (Z axis in scene space)
    const cosA = Math.cos(eyeOffset);
    const sinA = Math.sin(eyeOffset);
    const rx = dx * cosA - dy * sinA;
    const ry = dx * sinA + dy * cosA;

    return {
        eye: { x: center.x + rx, y: center.y + ry, z: eye.z },
        center: { ...center },
        up: { ...up },
    };
}

function camerasEqual(a, b) {
    if (!a || !b) return false;
    const eps = 1e-6;
    return Math.abs(a.eye.x - b.eye.x) < eps &&
           Math.abs(a.eye.y - b.eye.y) < eps &&
           Math.abs(a.eye.z - b.eye.z) < eps &&
           Math.abs(a.center.x - b.center.x) < eps &&
           Math.abs(a.center.y - b.center.y) < eps &&
           Math.abs(a.center.z - b.center.z) < eps &&
           Math.abs(a.up.x - b.up.x) < eps &&
           Math.abs(a.up.y - b.up.y) < eps &&
           Math.abs(a.up.z - b.up.z) < eps;
}

function renderStereo() {
    if (!stereoTraces || stereoTraces.length === 0) return;

    const sep = getStereoSeparation();
    const method = getStereoMethod();

    // Get current camera from mono plot if it exists
    const monoPlot = document.getElementById("embeddingPlot");
    let baseCamera = { eye: { x: 1.25, y: 1.25, z: 1.25 }, center: { x: 0, y: 0, z: 0 }, up: { x: 0, y: 0, z: 1 } };
    if (lastMonoCamera) baseCamera = lastMonoCamera;
    else if (monoPlot && monoPlot.layout && monoPlot.layout.scene && monoPlot.layout.scene.camera) {
        baseCamera = monoPlot.layout.scene.camera;
    }

    // Cross-eye: left panel shows RIGHT eye view, right panel shows LEFT eye view
    // Parallel: left panel shows LEFT eye view, right panel shows RIGHT eye view
    const sign = method === "cross" ? 1 : -1;
    const camL = offsetCamera(baseCamera, sign * sep);
    const camR = offsetCamera(baseCamera, -sign * sep);

    const layoutL = JSON.parse(JSON.stringify(EMBEDDING_LAYOUT));
    const layoutR = JSON.parse(JSON.stringify(EMBEDDING_LAYOUT));
    layoutL.scene.camera = camL;
    layoutR.scene.camera = camR;
    layoutL.showlegend = false;
    layoutR.showlegend = false;
    layoutL.margin = { l: 0, r: 0, t: 0, b: 0 };
    layoutR.margin = { l: 0, r: 0, t: 0, b: 0 };

    const stereoConfig = { ...EMBEDDING_CONFIG, displayModeBar: false };

    Plotly.newPlot("stereoPlotL", stereoTraces, layoutL, stereoConfig);
    Plotly.newPlot("stereoPlotR", stereoTraces, layoutR, stereoConfig);

    // Click handler for similar games
    [document.getElementById("stereoPlotL"), document.getElementById("stereoPlotR")].forEach(el => {
        el.on("plotly_click", function(eventData) {
            if (eventData.points.length > 0) {
                const battleId = eventData.points[0].customdata;
                if (battleId) fetchSimilar(battleId);
            }
        });
    });

    // Start polling-based camera sync
    startStereoSync();
}

function getGlCamera(plotEl) {
    // Reach into Plotly's internal GL scene to get the live camera
    // (updates during drag, not just on mouse-up like layout.scene.camera)
    try {
        const sceneKey = Object.keys(plotEl._fullLayout).find(k => k.startsWith("scene"));
        const scene = plotEl._fullLayout[sceneKey] && plotEl._fullLayout[sceneKey]._scene;
        if (scene && scene.glplot) {
            const gl = scene.glplot;
            return {
                eye: { x: gl.camera.eye[0], y: gl.camera.eye[1], z: gl.camera.eye[2] },
                center: { x: gl.camera.center[0], y: gl.camera.center[1], z: gl.camera.center[2] },
                up: { x: gl.camera.up[0], y: gl.camera.up[1], z: gl.camera.up[2] },
            };
        }
    } catch (e) { /* fall through */ }
    // Fallback to layout camera (only updates on mouse-up)
    return plotEl.layout && plotEl.layout.scene && plotEl.layout.scene.camera;
}

function setGlCamera(plotEl, cam) {
    // Directly set the GL camera for immediate visual update (no relayout needed)
    try {
        const sceneKey = Object.keys(plotEl._fullLayout).find(k => k.startsWith("scene"));
        const scene = plotEl._fullLayout[sceneKey] && plotEl._fullLayout[sceneKey]._scene;
        if (scene && scene.glplot) {
            const gl = scene.glplot;
            gl.camera.eye = [cam.eye.x, cam.eye.y, cam.eye.z];
            gl.camera.center = [cam.center.x, cam.center.y, cam.center.z];
            gl.camera.up = [cam.up.x, cam.up.y, cam.up.z];
            // Also update layout so it stays in sync when drag ends
            if (plotEl.layout && plotEl.layout.scene) {
                plotEl.layout.scene.camera = cam;
            }
            return true;
        }
    } catch (e) { /* fall through */ }
    return false;
}

function startStereoSync() {
    if (stereoPollId) cancelAnimationFrame(stereoPollId);

    let lastCamL = null;
    let lastCamR = null;

    function pollSync() {
        if (!stereoActive) return;

        const plotL = document.getElementById("stereoPlotL");
        const plotR = document.getElementById("stereoPlotR");
        if (!plotL || !plotR || !plotL._fullLayout) {
            stereoPollId = requestAnimationFrame(pollSync);
            return;
        }

        const camL = getGlCamera(plotL);
        const camR = getGlCamera(plotR);
        if (!camL || !camR) {
            stereoPollId = requestAnimationFrame(pollSync);
            return;
        }

        const sep = getStereoSeparation();
        const method = getStereoMethod();
        const sign = method === "cross" ? 1 : -1;

        // Detect which plot the user is dragging by comparing to last known state
        const lChanged = lastCamL && !camerasEqual(camL, lastCamL);
        const rChanged = lastCamR && !camerasEqual(camR, lastCamR);

        if (lChanged && !rChanged) {
            // User is dragging left plot — derive base camera, update right
            const baseCam = offsetCamera(camL, -(sign * sep));
            const newCamR = offsetCamera(baseCam, -(sign * sep));
            setGlCamera(plotR, newCamR);
            lastMonoCamera = baseCam;
            lastCamL = camL;
            lastCamR = newCamR;
        } else if (rChanged && !lChanged) {
            // User is dragging right plot — derive base camera, update left
            const baseCam = offsetCamera(camR, sign * sep);
            const newCamL = offsetCamera(baseCam, sign * sep);
            setGlCamera(plotL, newCamL);
            lastMonoCamera = baseCam;
            lastCamL = newCamL;
            lastCamR = camR;
        } else {
            lastCamL = camL;
            lastCamR = camR;
        }

        stereoPollId = requestAnimationFrame(pollSync);
    }

    stereoPollId = requestAnimationFrame(pollSync);
}

function stopStereoSync() {
    if (stereoPollId) {
        cancelAnimationFrame(stereoPollId);
        stereoPollId = null;
    }
}

function toggleStereo(enabled) {
    stereoActive = enabled;
    const monoPlot = document.getElementById("embeddingPlot");
    const stereoWrap = document.getElementById("stereoWrap");

    document.getElementById("stereo-method-label").style.display = enabled ? "" : "none";
    document.getElementById("stereo-sep-label").style.display = enabled ? "" : "none";

    if (enabled) {
        // Capture current camera before hiding
        if (monoPlot.layout && monoPlot.layout.scene && monoPlot.layout.scene.camera) {
            lastMonoCamera = monoPlot.layout.scene.camera;
        }

        // Build traces from current embedding data
        if (embeddingData && embeddingData.length > 0) {
            const corpusWins = embeddingData.filter(p => p.result === "win" && p.corpus !== "personal");
            const corpusLosses = embeddingData.filter(p => p.result === "loss" && p.corpus !== "personal");
            const personalWins = embeddingData.filter(p => p.result === "win" && p.corpus === "personal");
            const personalLosses = embeddingData.filter(p => p.result === "loss" && p.corpus === "personal");

            stereoTraces = [
                makeEmbeddingTrace(corpusWins,     "Corpus Wins",    "rgb(52, 211, 153)",  2.5, 0.25, false),
                makeEmbeddingTrace(corpusLosses,   "Corpus Losses",  "rgb(248, 113, 113)", 2.5, 0.25, false),
                makeEmbeddingTrace(personalWins,   "My Wins",        "rgb(52, 211, 153)",  2.5, 0.9,  true),
                makeEmbeddingTrace(personalLosses, "My Losses",      "rgb(248, 113, 113)", 2.5, 0.9,  true),
            ];
        }

        monoPlot.style.display = "none";
        stereoWrap.style.display = "";
        renderStereo();
    } else {
        stopStereoSync();
        stereoWrap.style.display = "none";
        monoPlot.style.display = "";
        // Restore camera to mono plot
        if (lastMonoCamera) {
            Plotly.relayout("embeddingPlot", { "scene.camera": lastMonoCamera });
        }
    }
}

// Wire up controls
document.getElementById("stereo-toggle").addEventListener("change", function() {
    toggleStereo(this.checked);
});
document.getElementById("stereo-method").addEventListener("change", function() {
    if (stereoActive) renderStereo();
});
document.getElementById("stereo-separation").addEventListener("input", function() {
    document.getElementById("stereo-sep-value").textContent = this.value;
    if (stereoActive) renderStereo();
});

async function fetchSimilar(battleId) {
    try {
        const resp = await fetch(`/api/similar/${battleId}`);
        if (!resp.ok) return;
        const data = await resp.json();
        renderSimilarPanel(battleId, data);
        drawSimilarityLines(data);
    } catch (e) {
        console.error("Failed to fetch similar games:", e);
    }
}

function drawSimilarityLines(data) {
    const ref = data.ref_coords;
    if (!ref || ref.x == null) return;

    const allSimilar = [...(data.personal || []), ...(data.corpus || [])];
    const valid = allSimilar.filter(s => s.x != null);
    if (valid.length === 0) return;

    const xs = [], ys = [], zs = [], texts = [];
    for (const s of valid) {
        xs.push(ref.x, s.x, null);
        ys.push(ref.y, s.y, null);
        zs.push(ref.z, s.z, null);
        texts.push("", `${s.opponent_name || "?"} (${s.result || "?"})`, "");
    }

    const lineTrace = {
        type: "scatter3d",
        mode: "lines",
        name: "Similar",
        x: xs, y: ys, z: zs,
        text: texts,
        hoverinfo: "text",
        showlegend: false,
        line: { color: "rgba(79, 140, 255, 0.5)", width: 1 },
    };

    const plotEl = document.getElementById("embeddingPlot");
    const nTraces = plotEl.data.length;

    if (nTraces > 4) {
        Plotly.deleteTraces("embeddingPlot", Array.from({length: nTraces - 4}, (_, i) => 4 + i));
    }

    Plotly.addTraces("embeddingPlot", lineTrace);
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
    document.getElementById("similar-ref").textContent = battleId.slice(0, 12) + "\u2026";

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

// ─── Win Probability (ADR-004) ───────────────────────────────────

function kebabToTitle(s) {
    return s.replace(/-/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

async function fetchWPCards() {
    try {
        const resp = await fetch("/api/wp/cards");
        if (!resp.ok) return;
        const data = await resp.json();
        renderWPCards(data);
    } catch (e) {
        // WP data not available yet
    }
}

function wpCardRows(cards) {
    return cards.map(c => {
        const netClass = c.net > 0 ? "wr-good" : c.net < 0 ? "wr-bad" : "";
        const netStr = c.net > 0 ? "+" + c.net : "" + c.net;
        return `<tr class="clickable" onclick="showCardArchetypes('${c.card}')" title="Click for archetype breakdown">
            <td>${kebabToTitle(c.card)}</td>
            <td>${c.carry}</td>
            <td>${c.liability}</td>
            <td>${c.critical}</td>
            <td class="${netClass}">${netStr}</td>
        </tr>`;
    }).join("");
}

function renderWPCards(data) {
    const section = document.getElementById("wp-section");
    section.style.display = "";

    document.getElementById("wp-total-games").textContent =
        `${data.total_games} games \u00b7 volatility ${data.avg_volatility.toFixed(4)}`;

    document.querySelector("#wp-team-table tbody").innerHTML =
        wpCardRows(data.team_cards || []);
    document.querySelector("#wp-opp-table tbody").innerHTML =
        wpCardRows(data.opp_cards || []);
}

async function showWPCurve(battleId) {
    if (!battleId) return;
    try {
        const resp = await fetch(`/api/wp/${battleId}`);
        if (!resp.ok) {
            document.getElementById("wp-game-label").textContent = "No WP data for this game";
            return;
        }
        const data = await resp.json();
        renderWPCurve(data);
    } catch (e) {
        console.error("Failed to fetch WP curve:", e);
    }
}

function renderWPCurve(data) {
    const section = document.getElementById("wp-section");
    section.style.display = "";

    const label = data.battle_id.length > 20
        ? data.battle_id.slice(0, 20) + "\u2026"
        : data.battle_id;
    document.getElementById("wp-game-label").textContent = label;

    const ctx = document.getElementById("wpChart").getContext("2d");
    if (wpChart) wpChart.destroy();

    const points = data.points;
    const critThreshold = 0.03;

    wpChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: points.map(p => p.tick),
            datasets: [{
                label: "P(win)",
                data: points.map(p => p.win_prob * 100),
                borderColor: "#4f8cff",
                borderWidth: 2,
                fill: false,
                tension: 0.1,
                pointRadius: points.map(p =>
                    Math.abs(p.wpa || 0) > critThreshold ? 5 : 0
                ),
                pointBackgroundColor: points.map(p => {
                    const wpa = p.wpa || 0;
                    if (wpa > critThreshold) return "#34d399";
                    if (wpa < -critThreshold) return "#f87171";
                    return "#4f8cff";
                }),
                pointBorderWidth: 0,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        title: (items) => `Tick ${points[items[0].dataIndex].tick}`,
                        afterLabel: (item) => {
                            const p = points[item.dataIndex];
                            const wpa = p.wpa || 0;
                            const sign = wpa >= 0 ? "+" : "";
                            return `WPA: ${sign}${(wpa * 100).toFixed(1)}%`;
                        },
                    },
                },
            },
            scales: {
                y: {
                    min: 0,
                    max: 100,
                    title: { display: true, text: "P(win) %" },
                    grid: {
                        color: (ctx) => ctx.tick.value === 50
                            ? "rgba(139, 143, 163, 0.5)"
                            : "rgba(42, 45, 58, 0.8)",
                    },
                },
                x: {
                    title: { display: true, text: "Game Tick" },
                    ticks: { maxTicksLimit: 10 },
                },
            },
        },
    });

    // Summary
    const s = data.summary;
    if (s) {
        const posCard = s.top_positive_wpa_card ? kebabToTitle(s.top_positive_wpa_card) : "—";
        const negCard = s.top_negative_wpa_card ? kebabToTitle(s.top_negative_wpa_card) : "—";
        document.getElementById("wp-summary").innerHTML =
            `<div class="wp-summary-row">
                <span>Start: ${(s.pre_game_wp * 100).toFixed(1)}%</span>
                <span>Final: ${(s.final_wp * 100).toFixed(1)}%</span>
                <span>Range: ${(s.min_wp * 100).toFixed(1)}–${(s.max_wp * 100).toFixed(1)}%</span>
                <span>Vol: ${s.volatility.toFixed(4)}</span>
            </div>
            <div class="wp-summary-row">
                <span class="wr-good">Carry: ${posCard}</span>
                <span class="wr-bad">Liability: ${negCard}</span>
            </div>`;
    }

    // Critical plays
    const crits = points
        .filter(p => Math.abs(p.wpa || 0) > critThreshold)
        .sort((a, b) => Math.abs(b.wpa) - Math.abs(a.wpa))
        .slice(0, 5);

    if (crits.length > 0) {
        document.getElementById("wp-critical-plays").innerHTML =
            `<div class="wp-crits"><strong>Top Swings:</strong> ` +
            crits.map(c => {
                const sign = c.wpa >= 0 ? "+" : "";
                const cls = c.wpa >= 0 ? "wr-good" : "wr-bad";
                return `<span class="${cls}">t${c.tick} ${sign}${(c.wpa * 100).toFixed(1)}%</span>`;
            }).join("  ") + `</div>`;
    } else {
        document.getElementById("wp-critical-plays").innerHTML = "";
    }
}

async function showCardArchetypes(cardName) {
    try {
        const resp = await fetch(`/api/wp/card/${cardName}`);
        if (!resp.ok) return;
        const data = await resp.json();

        const detail = document.getElementById("wp-archetype-detail");
        detail.style.display = "";
        document.getElementById("wp-arch-card").textContent = kebabToTitle(data.card);
        document.getElementById("wp-arch-side").textContent =
            `(${data.side === "team" ? "my card" : "opponent card"} \u00b7 ${data.total_games} games)`;

        const tbody = document.querySelector("#wp-arch-table tbody");
        tbody.innerHTML = data.archetypes.map(a => {
            const wpaClass = a.avg_wpa > 0.01 ? "wr-good" : a.avg_wpa < -0.01 ? "wr-bad" : "";
            const sign = a.avg_wpa >= 0 ? "+" : "";
            const totalSign = a.total_wpa >= 0 ? "+" : "";
            return `<tr>
                <td>${a.archetype}</td>
                <td>${a.games}</td>
                <td>${a.wins}</td>
                <td>${a.losses}</td>
                <td class="${wpaClass}">${sign}${(a.avg_wpa * 100).toFixed(1)}%</td>
                <td class="${wpaClass}">${totalSign}${(a.total_wpa * 100).toFixed(1)}%</td>
            </tr>`;
        }).join("");

        detail.scrollIntoView({ behavior: "smooth", block: "nearest" });
    } catch (e) {
        console.error("Failed to fetch card archetype detail:", e);
    }
}

// ─── Nemesis Dashboard ──────────────────────────────────────────

async function fetchNemeses() {
    try {
        sectionLoading("nemesis-section");
        const resp = await fetch("/api/nemeses");
        if (!resp.ok) { sectionReady("nemesis-section"); return; }
        const data = await resp.json();
        renderNemeses(data);
        sectionReady("nemesis-section");
    } catch (e) {
        sectionReady("nemesis-section");
    }
}

function renderNemeses(data) {
    if (!data || data.length === 0) return;

    const section = document.getElementById("nemesis-section");
    section.style.display = "";

    document.getElementById("nemesis-count").textContent =
        `${data.length} recurring opponents`;

    const tbody = document.querySelector("#nemesis-table tbody");
    tbody.innerHTML = data.map(n => {
        const tag = encodeURIComponent(n.opponent_tag);
        const wrCls = wrClass(n.win_rate);
        return `<tr class="clickable" onclick="showNemesisDetail('${tag}', '${(n.opponent_name || "Unknown").replace(/'/g, "\\'")}', this)" title="Click for weaknesses">
            <td>${n.opponent_name || "Unknown"}</td>
            <td>${n.times_faced}</td>
            <td>${n.wins}</td>
            <td>${n.losses}</td>
            <td class="${wrCls}">${n.win_rate}%</td>
            <td class="archetype-col">${n.archetype}</td>
        </tr>`;
    }).join("");
}

async function showNemesisDetail(encodedTag, name, rowEl) {
    // Highlight active row
    document.querySelectorAll("#nemesis-table tbody tr").forEach(tr =>
        tr.classList.remove("nemesis-active")
    );
    if (rowEl) rowEl.classList.add("nemesis-active");

    const placeholder = document.getElementById("nemesis-placeholder");
    const content = document.getElementById("nemesis-detail-content");
    placeholder.textContent = `Loading ${name}...`;
    placeholder.style.display = "";
    content.style.display = "none";

    try {
        const resp = await fetch(`/api/nemesis/${encodedTag}`);
        if (!resp.ok) {
            placeholder.textContent = "Failed to load nemesis data";
            return;
        }
        const data = await resp.json();

        placeholder.style.display = "none";
        content.style.display = "";

        document.getElementById("nemesis-detail-name").textContent = name;

        // Weaknesses panel
        const noCorp = document.getElementById("nemesis-no-corpus");
        const weakTbody = document.querySelector("#nemesis-weakness-table tbody");
        const weakTable = document.getElementById("nemesis-weakness-table");

        if (data.weakness_corpus_size < 5 || data.weaknesses.length === 0) {
            noCorp.style.display = "";
            weakTable.style.display = "none";
            document.getElementById("nemesis-corpus-size").textContent = "";
        } else {
            noCorp.style.display = "none";
            weakTable.style.display = "";
            document.getElementById("nemesis-corpus-size").textContent =
                `(${data.weakness_corpus_size} corpus games)`;
            weakTbody.innerHTML = data.weaknesses.slice(0, 15).map(w => {
                const pct = (w.posterior_mean * 100).toFixed(1);
                const ci = `[${(w.ci_low * 100).toFixed(0)}, ${(w.ci_high * 100).toFixed(0)}]`;
                return `<tr>
                    <td>${w.archetype}</td>
                    <td class="${wrClass(w.posterior_mean * 100)}">${pct}%</td>
                    <td>${w.wins}</td>
                    <td>${w.losses}</td>
                    <td class="ci-col">${ci}</td>
                </tr>`;
            }).join("");
        }

        // H2H panel
        const h2hTbody = document.querySelector("#nemesis-h2h-table tbody");
        h2hTbody.innerHTML = data.my_matchups.map(m => {
            return `<tr>
                <td>${m.archetype}</td>
                <td>${m.wins}</td>
                <td>${m.losses}</td>
                <td class="${wrClass(m.win_rate)}">${m.win_rate}%</td>
            </tr>`;
        }).join("");

    } catch (e) {
        placeholder.textContent = "Error loading nemesis data";
        console.error("Nemesis detail fetch failed:", e);
    }
}

// ─── Init & poll ────────────────────────────────────────────────

fetchAll();
fetchTilt();
fetchSimulation();
fetchWPCards();
fetchNemeses();
fetchEmbeddings();
setInterval(fetchAll, POLL_INTERVAL);
setInterval(fetchTilt, POLL_INTERVAL);
setInterval(fetchSimulation, POLL_INTERVAL);
setInterval(fetchNemeses, POLL_INTERVAL);
