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

## Materials intelligence platform

The backend is deliberately split into two planes:

1. the private lab plane retains canonical source snapshots, assessments, provenance, dataset governance, and the complete
   cross-client knowledge base;
2. isolated client workspaces receive only briefs matching their active research topics through
   `GET /v1/client-portal/{slug}/briefs`.

Client portal tokens are stored only as SHA-256 digests. Operational creation and configuration remain protected by the
backend service token. A research cycle prioritizes active client questions, falls back to the general lab agenda when
there are no clients, and publishes an idempotent brief only when topic keywords match the evidence.

Create a workspace and topic:

```bash
curl -X POST "$API/v1/client-workspaces" \
  -H "Authorization: Bearer $BACKEND_SERVICE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name":"Example Materials Co","slug":"example-materials","portal_token":"replace-with-32+-random-characters"}'

curl -X POST "$API/v1/client-workspaces/<workspace-id>/topics" \
  -H "Authorization: Bearer $BACKEND_SERVICE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name":"Membrane fouling","research_question":"Identify scalable hollow-fiber membrane treatments that reduce irreversible fouling without sacrificing permeability.","keywords":["membrane","fouling","permeability","hollow fiber"]}'
```

## Governed data federation

`python -m scripts.sync_dataset_registry` seeds the core catalog and discovers live OPTIMADE providers. The scheduled
workflow refreshes this registry before research. The registry records access tier, capabilities, licensing caveats, and
redistribution policy so that public metadata is not confused with licensed full text or proprietary property data.

Initial federation:

- OpenAlex and Crossref: scholarly literature and provenance;
- DataCite: experimental datasets and associated DOI metadata;
- OPTIMADE: normalized discovery across Materials Project, NOMAD, OQMD, JARVIS, AFLOW, Materials Cloud,
  Crystallography Open Database, and other providers;
- NIST Materials Data Repository: experimental and reference materials data;
- PubChem: structures, identifiers, properties, hazards, and chemistry links;
- DOE OSTI: energy and materials reports, software, and datasets;
- USPTO PatentsView: patent, inventor, assignee, and citation intelligence.

The platform stores provider metadata by default. Payload ingestion is enabled only when provider and record-level terms
permit it. Licensed sources such as commercial standards, handbooks, full-text journals, CSD/ICSD, and commercial patent
analytics require separate customer-funded agreements and must not be scraped into the shared lab.
