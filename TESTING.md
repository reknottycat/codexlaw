# Testing

## Deterministic suite

The local suite uses Python's standard `unittest` runner and never calls an external model:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

Run the same command on Linux or macOS with `PYTHONPATH=src`.

## Core acceptance checks

Validate the Python sources and Compose file without starting services:

```powershell
$env:PYTHONPATH = "src"
python -m compileall -q src scripts tests
$env:NEO4J_PASSWORD = "local-test-password"
docker compose config --quiet
```

The `scripts/dgx_acceptance.sh` entry point additionally creates an environment, installs the package, and runs the deterministic suite on Linux/DGX hosts.

## Real data preparation

All downloads stay under ignored `data/` paths. The current normalized case stream is recorded in `data/processed/benchmark/manifest.json`; it contains LegalBench, LegalBench-RAG, and CaseHOLD. The authority stream is recorded in `data/processed/authority/manifest.json`.

```powershell
$env:PYTHONPATH = "src"
bash scripts/bootstrap_sources.sh  # Git Bash; one-time Lawgent/Codex source checkout
python scripts/extract_legalbench_rag.py
python scripts/prepare_benchmark.py
python scripts/prepare_authority.py data/sources/ecfr/title-16-2026-08-31.xml data/sources/ecfr/title-17-2026-08-17.xml
```

The LegalBench-RAG archive contains filenames that Windows cannot create directly. `extract_legalbench_rag.py` writes a safe-path map and preserves the original archive paths in that map.

## Neo4j authority graph

Set a local `NEO4J_PASSWORD` only in the process environment, then start Neo4j and import the authority JSONL. The importer creates the `authority_id_unique` constraint, writes in batches, and rejects records whose source SHA-256 is not present in the authority manifest.

```powershell
$env:NEO4J_PASSWORD = "use-a-local-password"
docker compose up -d neo4j
python scripts/ingest_neo4j.py data/processed/authority/authority.jsonl `
  --uri bolt://localhost:7687 --user neo4j `
  --manifest data/processed/authority/manifest.json --replace-source
```

`--replace-source` keeps the graph aligned with the supplied authority manifest by removing only older versions whose `source_url` belongs to that manifest. Keep the password in the environment or an ignored password file; passing it as a command-line argument can expose it in process listings or shell history.

The hosted NVIDIA endpoint is allowlisted. If a private or self-hosted chat or embedding endpoint is intentional, set `NVIDIA_ALLOW_CUSTOM_ENDPOINT=true` and validate that the endpoint is trusted before providing a key. URLs containing userinfo, query strings, or fragments are rejected.

## DGX Spark embedding index

The default private endpoint is the verified DGX Spark 1B BF16 service. It is anonymous on the private network, so no NVIDIA key is used for embedding. The index includes the eCFR authority records and LegalBench-RAG corpus chunks.

```powershell
$env:PYTHONPATH = "src"
python scripts/build_embedding_index.py --batch-size 32
python scripts/query_embedding_index.py "What is the expiration date of this contract?"
```

## A/B benchmark on real cases

The runner selects a deterministic, source-balanced sample and executes A then B serially with the same K3 settings. The default effective output budget is 65536 tokens, the NVIDIA K3 API maximum, with `reasoning_effort=max`, streaming enabled, a 900-second request timeout, and a 30-second inter-request interval. Each completed A/B row is appended immediately under ignored `benchmark/results/` and reported on the console, so partial real-run evidence remains available if a later provider request is slow or fails.

```powershell
$env:PYTHONPATH = "src"
$env:LEGALBENCH_LIVE_CONFIRM = "true"
python scripts/run_ab_benchmark.py --limit 6 --source legalbench-rag,legalbench,casehold
```

Use a smaller `--limit` for a connectivity smoke test. A live run is intentionally not a load test or an instruction to send all 140,292 cases to a paid provider.

For a bounded comparative batch, keep the 65536-token default and make the retry envelope explicit: `NVIDIA_REQUEST_TIMEOUT_SECONDS=300` and `NVIDIA_RETRIES=1` mean at most two 300-second attempts per architecture/case. To apply a later evaluator improvement without sending another provider request, use `python scripts/rescore_ab_results.py benchmark/results/<run>.jsonl`; it only prints recomputed metrics and does not modify the original rows.

## Project-to-project benchmark

Install Python dependencies with `python -m pip install -e . -r requirements.txt`. This Windows entry point expects configured `mmx.cmd` and `codex.cmd` executables, plus the isolated Lawgent environment under `.runtime/lawgent-venv/`. MiniMax authentication uses the existing MMX configuration; no key is copied into benchmark records.

`run_project_ab_benchmark.py --codex-engine cli` executes the actual Codex CLI against Lawgent's `WorkflowExecutor`. Both receive the same frozen user task and evidence, MiniMax-M3, temperature 0.1, 4096 output tokens per call and a common per-case process-tree deadline. Case order alternates A/B. Workflow-specific system instructions and call counts remain different and are recorded. The text-only local Responses adapter reuses the same MiniMax Messages client; external tools are disabled. This does not test a general autonomous tool loop. The default `--codex-engine python` retains the Python-wrapper baseline.

Each run records input snapshots, SHA-256, code files including uncommitted edits, CLI version, raw and normalized answers, events, actual model usage and stop reasons. Historical metadata is reconstructed separately without rewriting old results or inventing unknown token counts. The evaluator is unchanged. See [the current audit](docs/project-status-2026-09-06.md).

The Lawgent runner uses the ignored `.runtime/lawgent-venv/` environment and never modifies `vendor/lawgent/`. On this Windows host, its PDF-export import is disabled only in the text benchmark subprocess because the unavailable GTK/Pango DLLs are irrelevant to a JSON legal-answer evaluation.

```powershell
$env:PYTHONPATH = "src"
python scripts/run_project_ab_benchmark.py --limit 60 --source legalbench-rag,legalbench,casehold --timeout 180 --codex-engine cli
```

The documented baseline uses `--limit 60`: 20 deterministic cases from each source. The batch is source-balanced; larger batches should keep the same model, closed-book policy, answer-type output contract, evaluator, and timeout.

### Offline vector-RAG variant

For an offline retrieval comparison, first prepare a separate, ignored case stream. It obtains embeddings from the configured private DGX endpoint and searches the local embedding index; it does not fetch legal material from the Internet. Retrieval evidence is appended after the existing benchmark evidence, so the evaluator can verify every cited evidence ID.

```powershell
$env:PYTHONPATH = "src"
python scripts/prepare_offline_rag_cases.py `
  --limit 60 --source legalbench-rag --source legalbench --source casehold `
  --output benchmark/results/offline-rag-cases.jsonl
python scripts/run_project_ab_benchmark.py `
  --cases benchmark/results/offline-rag-cases.jsonl --limit 60 --timeout 300
```

This command prepares vector evidence only. The corrected index uses `passage: ` for documents and `query: ` for queries, as required by the deployed Nemotron model. Legacy unprefixed indexes are rejected. Rebuild to a new path with `python scripts/reembed_index.py`; preserve the legacy index for historical reproduction.

### Live graph expansion and native Codex

The verified deployment is Neo4j 5.26.30 on DGX Spark, `bolt://192.168.1.6:7687`, in the independent `codelaw-neo4j` container. Configure `NEO4J_URI`, `NEO4J_USER` and `NEO4J_PASSWORD`, or use the ignored `.runtime/graph-connection.json` with a `password_file`. Never commit credentials. Graph expansion verifies every seed and source hash; missing graph data raises an error instead of silently falling back. REFERENCES edges come from explicit section references and SAME_PART denotes structural proximity, not semantic support.

```powershell
python scripts/prepare_graph_rag_cases.py --cases benchmark/results/offline-rag-cases.jsonl --output benchmark/results/vector-graph-cases.jsonl
python scripts/run_project_ab_benchmark.py --cases benchmark/results/vector-graph-cases.jsonl --limit 60 --timeout 180 --codex-engine cli
python scripts/reconstruct_run_metadata.py
python scripts/build_dashboard.py
```

Graph queries occur during preparation. Both harnesses then consume the exact same frozen retrieval results, rather than independently issuing live graph queries. The dashboard is a standalone HTML file and can be shared offline.

## Live NVIDIA Kimi K3 gate

The live gate uses NVIDIA's hosted OpenAI-compatible endpoint. It is strictly serial and requires both a runtime `NVIDIA_API_KEY` and explicit `LEGALBENCH_LIVE_CONFIRM=true`.

`run_k3_serial.py` is a single-architecture connectivity smoke over prepared real cases. For the actual comparison, use `run_ab_benchmark.py` below; it executes both A and B and writes evaluator metrics.

Check the effective configuration first:

```powershell
$env:PYTHONPATH = "src"
python scripts/check_config.py
```

The default output budget is 65536 tokens. Lower it for a short connectivity smoke test, or set `LIVE_LEGALBENCH_MAX_TOKENS=4096` when a faster, smaller evaluation is intentional. Never run the live gate as a load test.

```powershell
$env:PYTHONPATH = "src"
$env:LEGALBENCH_LIVE_CONFIRM = "true"
python scripts/run_k3_serial.py
```

Secrets must remain in the process environment and must not be committed.
