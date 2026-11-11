# Observability Stack

Start the local open-source observability stack:

```bash
docker compose -f deploy/observability/docker-compose.yml up -d
```

Configure the application process:

```env
OTEL_SERVICE_NAME=kbase-rag
OTEL_SERVICE_VERSION=0.1.0
OTEL_ENVIRONMENT=dev
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
OTEL_TRACES_SAMPLER=parentbased_traceidratio
OTEL_TRACES_SAMPLER_ARG=1.0
LOG_JSON_ENABLE=true

# LangGraph checkpoint persistence
LANGGRAPH_CHECKPOINT_BACKEND=postgres
LANGGRAPH_CHECKPOINT_NAMESPACE=kbase-rag
LANGGRAPH_CHECKPOINT_POSTGRES_DSN=postgresql://dataset_rag:dataset_rag_password@localhost:15432/dataset_rag_observability
LANGGRAPH_CHECKPOINT_POSTGRES_SETUP=true
LANGGRAPH_CHECKPOINT_POSTGRES_POOL_MIN=1
LANGGRAPH_CHECKPOINT_POSTGRES_POOL_MAX=10
LANGGRAPH_CHECKPOINT_RETENTION_DAYS=30
OBSERVABILITY_SNAPSHOT_LEVEL=safe
OBSERVABILITY_REPLAY_ALLOW_LIVE=false
```

Local endpoints:

- Grafana: http://localhost:13000, default account `admin/admin`
- Prometheus: http://localhost:19090
- Alertmanager: http://localhost:19093
- Loki: http://localhost:3100
- Tempo: http://localhost:3200
- OpenTelemetry Collector OTLP HTTP: http://localhost:4318
- LangGraph checkpoint Postgres: `localhost:15432`, database `dataset_rag_observability`

Alertmanager is provisioned as a Grafana data source and receives Prometheus
rules from `deploy/observability/rules/kbase-rag-alerts.yml`. The initial
rule set covers missing business metrics, OTel Collector availability, query
error rate and latency, import error rate and latency, and Milvus error rate
and latency. The current rule file also points each warning/critical alert to a
dedicated runbook section in `OBSERVABILITY_ALERTING_OPTIMIZATION.md`.

Grafana automatically loads these dashboards:

- `Dataset RAG Overview`
- `Dataset RAG Query`
- `Dataset RAG Import`
- `Dataset RAG Dependencies`
- `Dataset RAG Observability`

Recommended triage path:

1. Open the alert from Alertmanager or Grafana and jump to the linked dashboard.
2. Use the overview dashboard to decide whether this is a user-facing issue or a dependency issue.
3. Open the query/import/dependency dashboard that matches the alert.
4. Use `trace_id`, `session_id`, or `task_id` to open the matching timeline under `/debug`.
5. Re-run the failing node or workflow with `/debug/replay/node` or `/debug/replay/failed` if the failure looks deterministic.

Stability automation runs inside the query/import application process on
startup. It records lightweight readiness and observability probes, scans
processing tasks for long idle periods, and emits filesystem capacity metrics.
Useful settings:

```env
OBSERVABILITY_AUTOMATION_ENABLED=true
OBSERVABILITY_AUTOMATION_INTERVAL_SECONDS=60
OBSERVABILITY_QUERY_STUCK_SECONDS=600
OBSERVABILITY_IMPORT_STUCK_SECONDS=1800
OBSERVABILITY_STUCK_EVENT_COOLDOWN_SECONDS=300
OBSERVABILITY_CAPACITY_WARNING_PERCENT=85
OBSERVABILITY_CAPACITY_CRITICAL_PERCENT=95
OBSERVABILITY_CAPACITY_PATHS=/path/to/prometheus,/path/to/loki,/path/to/tempo
```

Backfill local JSONL fallback events into MongoDB after Mongo recovers:

```bash
python -m app.observability.jsonl_backfill
python -m app.observability.jsonl_backfill --write
```

Validate Prometheus rules before shipping changes:

```bash
python deploy/observability/validate_prometheus_rules.py
```

If `promtool` is installed, the script performs full Prometheus syntax checks;
otherwise it falls back to structural validation.

The default Alertmanager receivers send resolved and firing notifications to
`http://host.docker.internal:9000/alerts/{default,critical,warning}`. Replace
these webhook URLs with your IM, email, or incident-routing endpoint before
using the stack for production alert delivery.

Quick validation steps:

1. Start the stack with `docker compose -f deploy/observability/docker-compose.yml up -d`.
2. Stop `otel-collector` and confirm `OtelCollectorDown` becomes critical after
   2 minutes in Alertmanager.
3. Generate failed query requests until `rag_query_total{status="error"}` rises
   and confirm `QueryHighErrorRate` fires after 5 minutes.
4. Restore the failing component and confirm Alertmanager sends a resolved
   notification.

The application writes JSON logs to `logs/app_YYYYMMDD.jsonl`; Promtail mounts
that directory and forwards the logs to Loki.

When `LANGGRAPH_CHECKPOINT_BACKEND=postgres`, the application initializes the
LangGraph checkpoint tables on startup if `LANGGRAPH_CHECKPOINT_POSTGRES_SETUP`
is true. In production, keep this backend enabled and avoid falling back to
memory, otherwise workflow replay checkpoints will be lost after restart.

For production replay, keep `OBSERVABILITY_REPLAY_ALLOW_LIVE=false` by default
and use `mock_external` for incident diagnosis. Enable live replay only in
controlled or isolated environments because it may call LLM/Web/Milvus and other
side-effecting dependencies.
