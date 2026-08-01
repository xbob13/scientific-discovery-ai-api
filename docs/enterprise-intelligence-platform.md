# Enterprise intelligence operating model

## Product boundary

Patterson Research Labs sells governed research intelligence—not raw copyrighted content and not autonomous scientific
discovery. A client purchases a bounded or continuous mandate. The platform retrieves permitted metadata, preserves
provenance, proposes reviewable research leads, maps patent signals, and returns a client-isolated evidence package.

## Runtime flow

1. Hosted checkout verifies payment outside the browser.
2. The backend idempotently provisions the workspace, entitlement, topic, and first mandate.
3. A background task claims the mandate atomically; the durable worker recovers anything left queued.
4. Literature adapters harvest OpenAlex, Crossref, and DataCite records.
5. Patent adapters query USPTO Open Data Portal and EPO OPS; available claims become source-linked technical evidence.
6. Canonical records, raw snapshots, candidates, patent metadata, checksums, and limitations are retained.
7. A client-specific brief is published only to the originating workspace.
8. The portal shows live mandate state and published evidence packages.

## Plan enforcement

| Plan | Rolling request allowance | Source policy | Exports |
| --- | ---: | --- | --- |
| Commissioned brief | 3 | OpenAlex, Crossref, DataCite, USPTO ODP | JSON, Markdown |
| Continuous monitor | 30 | Literature plus USPTO ODP and EPO OPS | JSON, Markdown |
| Enterprise | 250 | Full configured governed set | JSON, Markdown, CSV |

The policy lives in `research_lab/commerce.py` and is copied to a workspace entitlement at activation. Changing a global
plan does not silently rewrite an existing commercial entitlement.

## Source governance

Every source registry item carries its access tier, capability list, license summary, and redistribution policy. Metadata
availability must not be interpreted as permission to redistribute full text. Commercial journals, standards, proprietary
property databases, and licensed patent analytics require a separate agreement and an explicit tenant-specific policy.

WIPO PATENTSCOPE public search is not treated as a scrape target. Add WIPO only through an authorized data product or feed.
Provider quotas, robots policies, and contractual restrictions remain hard boundaries.

## Outreach boundary

Account research, policy qualification, and delivery may be automated from public evidence:

`researched account -> evidence/domain/sender policy -> autonomous queue -> provider delivery -> audit result`

Suppressed accounts cannot be queued or delivered. Both autonomous and delivery switches default to false. The queue
requires official evidence, a documented contact basis, organization-domain matching, a non-consumer address, sender
identity, physical address, signed unsubscribe, and daily caps. Missing any gate holds the message automatically.

## Production readiness checklist

- Use PostgreSQL with encrypted backups and tested restoration.
- Rotate backend, portal-signing, patent-provider, Stripe, and sender credentials independently.
- Run Alembic through `0005_autonomous_network` before accepting checkout traffic.
- Configure Stripe price IDs, the signed webhook secret, and all lifecycle events before enabling checkout.
- Run a durable mandate worker; keep the GitHub workflow as recovery, not as an SLA scheduler.
- Set USPTO ODP/EPO credentials and verify provider quota behavior.
- Keep autonomous outreach disabled until sender identity, suppression, unsubscribe, daily caps, and applicable jurisdiction policies are configured.
- Add observability for queue age, failed mandates, provider failures, checkout activation failures, and delivery failures.
- Benchmark citation correctness, novelty classification, and expert acceptance before making performance claims.
