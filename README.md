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
