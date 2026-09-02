# CodexLaw real-data acceptance snapshot

Generated on 2026-09-03 from commit `6ab42ce`. This is a dated acceptance snapshot; the manifests and JSONL result files are the authoritative artifacts for reproduction.

## Prepared data

- `data/processed/benchmark/manifest.json`: 140,292 normalized public cases from LegalBench-RAG, LegalBench, and CaseHOLD.
- `data/processed/authority/manifest.json`: 5,683 official eCFR Title 16/17 records with source URL, issue date, original text, and SHA-256.
- `data/processed/embedding-index/manifest.json`: 58,604 chunks, 2,048 dimensions, produced by the verified DGX Spark `nvidia/Nemotron-3-Embed-1B-BF16` endpoint.

All three data artifacts are intentionally ignored by Git because they are large local rebuild outputs. `data/README.md` records the public sources and license boundaries.

## Runtime checks

- Effective chat model: `moonshotai/kimi-k3` on NVIDIA's OpenAI-compatible endpoint.
- Effective K3 budget: 20,000 output tokens; API validation permits 1–65,536, so 140,096 is invalid.
- Effective K3 settings: `temperature=1`, `reasoning_effort=max`, streaming enabled, 900-second request timeout, default retries `2`, and 30-second serial spacing.
- DGX embedding returned 2,048-dimensional vectors; the full index authority hash matches the current authority JSONL.
- Neo4j contains 5,683 authority nodes across 2 issue dates and has the `authority_id_unique` uniqueness constraint.
- Prometheus readiness endpoint returned HTTP 200.
- Standard wheel build succeeded for `codelaw-0.2.0`.
- Deterministic suite: 60 tests passed, including a `ResourceWarning`-as-error run; `compileall`, Compose validation, and staged secret scan passed.

## Real K3 A/B sample

The runner used the same cases, evidence, evaluator, and K3 settings for Architecture A (Lawgent) and B (CodexLaw workflow). The paid-provider runs explicitly set `NVIDIA_RETRIES=0` to avoid duplicate billable requests; the stored default remains `2`.

- LegalBench-RAG: A succeeded 1/1; B succeeded 0/1 under the strict evidence-span metric. B returned a semantically correct paraphrase, with valid citation and workflow compliance.
- LegalBench plus CaseHOLD: A succeeded 2/2 and B succeeded 2/2. Citation validity and workflow compliance were 100% for all four rows.

The raw evidence-bearing rows and summaries are under ignored `benchmark/results/`. Re-run a larger, source-balanced sample with `scripts/run_ab_benchmark.py --limit N --source legalbench-rag,legalbench,casehold` after setting the runtime key and explicit live confirmation.

No API key is stored in the repository or in this report.
