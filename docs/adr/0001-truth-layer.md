# ADR 0001: Canonical truth layer and bounded workflows

Status: accepted (initial vertical slice)

Exponent Bio remains the authenticated Base44 control plane. This service becomes the canonical scientific store.
PostgreSQL is the production database; SQLite is supported only for deterministic local tests. Source adapters retain
immutable checksummed payloads before canonicalization. DOI normalization merges works but never deletes source versions.

The first slice uses OpenAlex and Crossref because they provide official, legally accessible scholarly metadata and overlap
for deduplication tests. It is deliberately non-medical. An analytical conductor-resistance Monte Carlo model provides the
first real compute artifact; it is labeled as a low-fidelity model, seeded, hashed, and replay-checked.

No LLM output can become verified evidence by itself. Future extractors must create claims with stored work-version locators,
and future connection promotion requires proposer, skeptic, and citation-verifier stages. Base44 receives projections only.
