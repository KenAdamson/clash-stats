# Grafana contributions (dashboards/ + provisioning/)

Dashboards contributed to the shared observability stack (server-docs owns the
stack; this dir is bind-mounted to /etc/grafana/dashboards/clash per the
three-mount contract). JSON dashboards only — providers are owned by
server-docs, not here.

The pre-extraction stack (and its provisioning/dashboards/ dir with the
legacy provider yamls) was deleted at the 2026-07-28 cutover tidy-up;
dashboards/ here is the single source of truth.
