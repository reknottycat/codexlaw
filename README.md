# CodexLaw

A reproducible Legal Agent Harness A/B project.

## Goal

Compare:

- **Architecture A — Lawgent Native**
- **Architecture B — Codex Core + Legal Domain Pack**

The project will benchmark workflow control, retrieval, citation verification, recovery, context preservation, and evidence handling on the same lawful test data.

## Recovery status

This repository was initialized on 2026-09-01 after the earlier transient workspace was automatically removed. The previous custom harness and run artifacts are not being represented as recovered source. Reimplementation starts from this commit and every subsequent phase will be committed here.

## Safety and data rules

- API keys are read only from runtime environment variables and are never committed.
- LegalBench is evaluation data, not legal authority.
- Legal conclusions must remain grounded in primary legal sources and independently verified.
- Official source material keeps provenance, version/effective-date metadata, and original text.
- No commercial or access-restricted legal database is copied into this repository.

## Planned layout

```text
architecture-a-lawgent/
architecture-b-codex/
shared/
  retrieval/
  verification/
  evidence/
benchmark/
  datasets/
  runner/
  results/
observability/
scripts/
docs/
```

## First reproducibility target

1. Rebuild the shared data and Neo4j pipeline.
2. Rebuild the two harness adapters without dual agent runtimes.
3. Add deterministic workflow and evidence-ledger tests.
4. Run a small, strictly serial K3 quality gate only when an environment-provided API key and network access are available.
