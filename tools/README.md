# Ops toolbox

Training launchers, evaluation harnesses, diagnostics and feature research that
support the ML pipeline. Baked into the image at `/app/tools/` so the container
is self-contained — the scripts that build and evaluate the models ship with the
runtime that uses them.

These previously lived loose in the `./data` volume. That directory is a symlink
to storage outside the repo, and git does not traverse symlinks, so nothing in
there was ever under source control and every script had to be `docker cp`'d in
by hand.

## Running them

**Run with `cwd=/app`.** Several scripts resolve model and shard paths relative
to the working directory (e.g. `data/ml_models/wp_v*.pt`), so they break if run
from elsewhere:

```sh
docker exec cr-tracker bash -c 'cd /app && PYTHONPATH=/app/src python3 tools/eval/eval_conf_counts.py'
docker exec -d cr-tracker bash /app/tools/launchers/wp_capacityB_resume.sh
```

`PYTHONPATH=/app/src` runs against the working tree. Without it, `import tracker`
resolves to the *installed* package in `site-packages` — which is what the
`clash-stats` CLI uses, and the two can drift if only one has been updated.

Data (checkpoints, shards, logs) stays in `/app/data/` — the volume. Only the
code moved.

## Layout

| Path | Contents |
|------|----------|
| `build_wp_shards.py` | Pre-extract the WP memmap shard cache (the "firehose" input) |
| `launchers/` | Long-running training launchers. All take `xpu_train.lock` — the A770 cannot hold two training jobs, and a cron retrain once evicted a 25-epoch run |
| `eval/` | Calibration-by-tier, confident-subset coverage, error critics, physics validation |
| `diag/` | Smoke tests, NaN hunts, batch-size probes, memory probes. Mostly single-purpose, kept because they encode hard-won reproductions |
| `analysis/` | Feature research: predictive placement detection, elixir-advantage signal, war-deck selection |

## Conventions

- Launchers write to `/app/data/<name>.log` and are started detached
  (`docker exec -d`); training runs outlast any shell.
- Anything touching the XPU for **training** must hold `xpu_train.lock`.
  Inference deliberately does not — see ADR-010 and `entrypoint.sh`.
- Long runs checkpoint best-weights only (no optimizer state). Use `WP_RESUME`
  to warm-start after an interrupted run, paired with a reduced `WP_LR`.
- An aborted training run leaves an **unregistered** `wp_vN.pt`; the next run
  reuses `N` and overwrites it. Copy it to a descriptive name first.
