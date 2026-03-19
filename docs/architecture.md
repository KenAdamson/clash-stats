# Architecture

## System Overview

```mermaid
graph TB
    subgraph Internet
        CR_API[Clash Royale API<br/>api.clashroyale.com]
        ROYALE[RoyaleAPI<br/>royaleapi.com<br/>replay HTML]
        GH[GitHub<br/>stats branch]
    end

    subgraph "Docker Compose Stack (192.168.7.58)"
        subgraph "Core Services"
            TRACKER[cr-tracker<br/>Python 3.11<br/>Flask dashboard :8078<br/>Prometheus :8001]
            PG[(PostgreSQL 16<br/>clash-postgres<br/>JSONB + TOAST)]
            BROWSER[cr-browser<br/>Headless Chromium<br/>NoVNC :6080]
        end

        subgraph "ML / Vision"
            SAMV2[cr-samv2<br/>SAMv2 Hiera Large<br/>Intel Arc A770 XPU<br/>:8079]
        end

        subgraph "Observability"
            PROM[Prometheus]
            LOKI[Loki]
            ALLOY[Alloy<br/>Docker log shipper]
            GRAFANA[Grafana<br/>:3000]
        end
    end

    subgraph "User"
        DASH[Web Dashboard<br/>Chart.js + Plotly.js]
        DISCORD[Discord<br/>Alerts]
    end

    CR_API -- "battles + profiles" --> TRACKER
    ROYALE -- "replay HTML" --> BROWSER
    BROWSER -- "parsed replays" --> TRACKER
    TRACKER -- "SQLAlchemy ORM" --> PG
    TRACKER -- "stats JSON" --> GH
    TRACKER -- "frame tracking" --> SAMV2
    TRACKER -- "/metrics" --> PROM
    PROM --> GRAFANA
    LOKI --> GRAFANA
    ALLOY -- "container logs" --> LOKI
    GRAFANA -- "webhook" --> DISCORD
    DASH -- ":8078" --> TRACKER

    style PG fill:#336791,color:#fff
    style TRACKER fill:#306998,color:#fff
    style SAMV2 fill:#0071c5,color:#fff
    style GRAFANA fill:#f46800,color:#fff
```

## Data Pipeline

```mermaid
flowchart LR
    subgraph "Ingest (every 1-2 min)"
        A1[CR API poll<br/>25 battles/player] --> A2[SHA-256 dedup]
        A2 --> A3[Store battle<br/>+ deck_cards]
        A3 --> A4[Replay HTTP fetch<br/>per player]
        A4 --> A5[Parse replay events<br/>+ summaries]
    end

    subgraph "Storage (PostgreSQL)"
        B1[(battles<br/>1.3M rows<br/>JSONB raw_json)]
        B2[(deck_cards<br/>24M rows)]
        B3[(replay_events<br/>8M rows)]
        B4[(player_corpus<br/>19K players)]
    end

    subgraph "ML Pipeline"
        C1[Feature extraction<br/>50-dim vectors]
        C2[UMAP 50→15→3<br/>+ HDBSCAN]
        C3[TCN Encoder<br/>256-dim embeddings]
        C4[Win Probability<br/>Platt-calibrated TCN]
        C5[Activity Model<br/>GBM classifier]
    end

    subgraph "Output"
        D1[Flask Dashboard<br/>Trophy charts<br/>3D manifold<br/>WP curves]
        D2[Prometheus Metrics<br/>Batch yields<br/>Scrape timing]
        D3[GitHub Stats<br/>JSON snapshots]
    end

    A3 --> B1
    A3 --> B2
    A5 --> B3
    B3 --> C1 --> C2
    C1 --> C3
    B3 --> C4
    B1 --> C5
    C5 -. "prioritize<br/>scrape queue" .-> A1
    B1 --> D1
    C2 --> D1
    C4 --> D1
    B1 --> D2
    B1 --> D3
```

## ML Models

```mermaid
graph LR
    subgraph "Training Data"
        RE[replay_events<br/>8M card placements]
        BT[battles<br/>1.3M games]
    end

    subgraph "Models"
        TCN[TCN Encoder<br/>tcn_v1.pt<br/>6-layer causal<br/>256-dim output]
        WP[Win Probability<br/>wp_v1.pt<br/>Causal TCN<br/>78.4% accuracy]
        CAL[Platt Calibrator<br/>wp_calibrator.json<br/>ECE = 0.031]
        ACT[Activity Model<br/>activity_model.pkl<br/>GBM classifier<br/>10 features]
    end

    subgraph "Inference"
        UMAP[UMAP + HDBSCAN<br/>3D manifold<br/>cluster profiles]
        WPI[Per-tick P_win<br/>WPA per card<br/>Critical moments]
        PRI[Corpus scheduling<br/>P_active scoring<br/>50-player batches]
    end

    RE --> TCN --> UMAP
    RE --> WP --> WPI
    WP --> CAL --> WPI
    BT --> ACT --> PRI

    style TCN fill:#306998,color:#fff
    style WP fill:#306998,color:#fff
    style ACT fill:#306998,color:#fff
```

## Cron Schedule

| Interval | Job | Description |
|----------|-----|-------------|
| */2 min | `personal_combined.sh` | Fetch personal battles + scrape replays |
| */1 min | `corpus_combined.sh` | 50-player batch: battles + replays (flock guard) |
| */5 min | `publish_wrapper.sh` | Push stats JSON to GitHub |
| */5 min | `wp_infer_new.sh` | Incremental WP inference on new replay games |
| Hourly | `corpus_nemeses.sh` | Add opponents from personal losses to corpus |
| Daily 3am | `corpus_discover.sh` | Network discovery from opponent tags |
| Weekly Mon 6am | `corpus_update.sh` | Refresh top-ladder player list |
| Weekly Mon 6:30am | `train_activity.sh` | Retrain activity model |
| Weekly Mon 7am | `corpus_locations.sh` | Regional leaderboard discovery |
| Weekly Sun 4am | `tcn_train.sh` | Retrain TCN encoder |
| 6-hourly | `sim_refresh.sh` | Refresh Monte Carlo simulation results |
