Dashboards contributed to the shared observability stack (server-docs owns the
stack; this dir is bind-mounted to /etc/grafana/dashboards/clash per the
three-mount contract). JSON dashboards only — providers are owned by
server-docs, not here.

The copy of pipeline.json in ../provisioning/dashboards/ is the LEGACY
location scanned by the pre-extraction stack; it is deleted at the cutover
tidy-up. Until then, edits go to BOTH copies or, better, wait for cutover.
