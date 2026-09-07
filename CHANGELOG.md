# Changelog

All notable changes to CodexLaw are documented here.

## [0.5.0.0] - 2026-09-07

### Added

- Run a project-to-project legal benchmark with the real Codex CLI through a bounded text-only Responses adapter, while recording input snapshots, source hashes, model usage, stop reasons, and execution events.
- Prepare source-balanced offline vector retrieval cases and expand them through Neo4j authority relationships with source-hash validation.
- Reconstruct historical run diagnostics without rewriting original results, and build a standalone offline dashboard for paired outcomes, failure categories, provenance, and token telemetry.
- Add deterministic tests for shared benchmark inputs, process-tree deadlines, provider failures, graph provenance, embedding prefixes, and project output contracts.

### Changed

- Keep Lawgent and CodexLaw on the same task, evidence, normalisation, output budget, temperature, and per-case deadline while preserving each workflow's own call count.
- Use the Nemotron query: and passage: input prefixes, reject legacy indexes, preserve supplied evidence before retrieval evidence, and keep the v1 index intact when rebuilding.
- Make MiniMax requests serial, retain JSON usage and stop-reason telemetry, and document the verified graph, embedding, native Codex, and dashboard workflows.

### Fixed

- Prevent malformed Lawgent workflow objects and bare labels from being mistaken for valid JSON answers, and separate answer, citation, formatting, and provider failures in the audit.
- Stop graph imports from accepting authority records outside the supplied source manifest, and stop embedding queries from silently using an incompatible model or legacy index.
- Bound child-process execution and report recoverable provider timeouts and HTTP failures without leaking credentials or traceback details.
