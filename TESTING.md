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
  --uri bolt://localhost:7687 --user neo4j --password $env:NEO4J_PASSWORD `
  --manifest data/processed/authority/manifest.json --replace-source
```

`--replace-source` keeps the graph aligned with the supplied authority manifest by removing only older source-hash versions.

## DGX Spark embedding index

The default private endpoint is the verified DGX Spark 1B BF16 service. It is anonymous on the private network, so no NVIDIA key is used for embedding. The index includes the eCFR authority records and LegalBench-RAG corpus chunks.

```powershell
$env:PYTHONPATH = "src"
python scripts/build_embedding_index.py --batch-size 32
python scripts/query_embedding_index.py "What is the expiration date of this contract?"
```

## A/B benchmark on real cases

The runner selects a deterministic, source-balanced sample and executes A then B serially with the same K3 settings. The default effective output budget is 20000 tokens, with `reasoning_effort=max`, streaming enabled, a 900-second request timeout, and a 30-second inter-request interval. Results are written under ignored `benchmark/results/`.

```powershell
$env:PYTHONPATH = "src"
$env:LEGALBENCH_LIVE_CONFIRM = "true"
python scripts/run_ab_benchmark.py --limit 6 --source legalbench-rag,legalbench,casehold
```

Use a smaller `--limit` for a connectivity smoke test. A live run is intentionally not a load test or an instruction to send all 140,292 cases to a paid provider.

## Live NVIDIA Kimi K3 gate

The live gate uses NVIDIA's hosted OpenAI-compatible endpoint. It is strictly serial and requires both a runtime `NVIDIA_API_KEY` and explicit `LEGALBENCH_LIVE_CONFIRM=true`.

`run_k3_serial.py` is a single-architecture connectivity smoke over prepared real cases. For the actual comparison, use `run_ab_benchmark.py` below; it executes both A and B and writes evaluator metrics.

Check the effective configuration first:

```powershell
$env:PYTHONPATH = "src"
python scripts/check_config.py
```

The default output budget is 20000 tokens. Lower it for a short connectivity smoke test, or set `LIVE_LEGALBENCH_MAX_TOKENS=4096` when a faster, smaller evaluation is intentional. Never run the live gate as a load test.

```powershell
$env:PYTHONPATH = "src"
$env:LEGALBENCH_LIVE_CONFIRM = "true"
python scripts/run_k3_serial.py
```

Secrets must remain in the process environment and must not be committed.
