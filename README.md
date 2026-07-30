# Patterson Research Labs research backend

Canonical, provenance-first backend for Exponent Bio. This replaces the former random sentence generator.

## Local verification

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements-dev.txt
pytest -q
alembic upgrade head
uvicorn research_lab.main:app --reload --port 10000
```

Use `docker compose up --build` for PostgreSQL-backed development. Copy `.env.example` to `.env` and replace all
development secrets. The API requires `Authorization: Bearer <BACKEND_SERVICE_TOKEN>` for operational endpoints.

The system produces research intelligence and unvalidated hypotheses. It does not provide medical advice, patentability
opinions, autonomous publications, filings, purchases, messages, or experiments.

## Autonomous digital lab

The lab runs as bounded research cycles rather than a collection of free-running chatbots. Each cycle:

1. retrieves records from configured primary scholarly indexes;
2. stores immutable, checksummed source snapshots;
3. canonicalizes works and versions;
4. proposes transparent cross-paper retrieval hypotheses;
5. records assumptions, falsifiable predictions, validation methods, and rejection criteria; and
6. emits inspectable JSON and Markdown lab reports with source links and exact evidence excerpts.

GitHub Actions is the initial zero-cost execution environment. It runs at minutes 7 and 37 of each hour, rotating through
an engineering, materials, energy, water, sensors, and manufacturing agenda. This cadence is designed to remain near the
2,000-minute monthly allowance of a private GitHub Free repository. The workflow is fail-closed until the repository
variable `RESEARCH_KILL_SWITCH=false` is deliberately configured. It uses a concurrency lock, a 20-minute ceiling,
least-privilege read access, cached SQLite state, and retained run artifacts. It does not require Base44.

Run a cycle locally with:

```bash
python scripts/run_research_cycle.py --question "your bounded research question"
```

The connection agent uses deterministic information-retrieval signals. Every candidate includes a decomposed
review-priority score, source evidence, falsifiable prediction, validation method, prior-art query, corroboration state,
limitations, and rejection criteria. A proposed connection is explicitly not a scientific conclusion and cannot promote
itself without corroborating evidence. Authenticated clients can inspect ranked assessments at `GET /v1/findings`.
