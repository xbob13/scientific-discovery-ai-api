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
opinions, autonomous publications, filings, purchases, or experiments. Outbound messages require named human approval,
and the delivery integration is disabled by default.

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

## Self-service intelligence platform

The API supports verified checkout activation without trusting browser-supplied payment state. A server-side commerce
function verifies the provider session and then calls `POST /v1/subscriptions/activate` using the backend service token.
Activation is idempotent and creates:

- a token-hashed, isolated client workspace;
- a source and export entitlement for the purchased plan;
- an initial client topic and metered research mandate; and
- an auditable subscription activation record.

Clients can submit further bounded questions at `POST /v1/client-portal/{slug}/mandates`. The API enforces the workspace
token, source entitlement, and rolling 30-day request allowance before accepting the job. Long-running backends begin the
job immediately as a background task. `scripts/run_pending_mandates.py` and the `client-mandates.yml` workflow provide a
durable recovery worker for queued jobs.

Configure a unique `PORTAL_TOKEN_SIGNING_SECRET`; do not reuse the backend service token. Production workers require a
durable PostgreSQL `PRODUCTION_DATABASE_URL`. GitHub schedule timing is best-effort, so a continuously running queue worker
or managed scheduled job is recommended for contractual processing deadlines.

## Patent intelligence

`POST /v1/patents/search` federates configured USPTO PatentsView and EPO OPS searches, normalizes records, retains raw
provider provenance and checksums, and returns an explainable landscape of assignees, classifications, publication years,
and query-term coverage. Add `PATENTSVIEW_API_KEY` and/or EPO OPS consumer credentials to activate providers.

Patent output is research triage. It is not a legal-status, patentability, validity, infringement, or freedom-to-operate
opinion. The application should link to the authoritative source record and use patent counsel for consequential IP
decisions.

## Governed commercial outreach

Prospects must be backed by a public source URL, a relevance signal, a business contact, and a documented contact basis.
The system can prepare a relevant first-touch draft, but it cannot send from draft state. A named administrator must
approve each message. Delivery also requires all of the following:

```env
OUTREACH_SEND_ENABLED=true
OUTREACH_DELIVERY_WEBHOOK_URL=https://approved-sender.example/send
OUTREACH_DELIVERY_TOKEN=replace-with-provider-token
```

Suppression immediately blocks pending messages. Provider delivery IDs, approver identity, approval time, delivery state,
and errors remain in the audit ledger. The operator is still responsible for applicable anti-spam, privacy, sender
identification, and unsubscribe requirements in every recipient jurisdiction.

An optional evidence-feed worker can ingest researched accounts and prepare drafts automatically. It accepts structured
records from an approved search/data provider; it does not scrape arbitrary sites. Set `PROSPECT_DISCOVERY_ENABLED=true`
and configure the feed URL/token to use it. The hourly commercial workflow can also deliver messages already in `approved`
state when `OUTREACH_SEND_ENABLED=true`. Discovery never approves or sends its own drafts.
