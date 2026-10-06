# Offline vector-RAG project A/B: MiniMax-M3

## Result

This completed batch compares the original Lawgent workflow with CodexLaw orchestration on 60 real, source-balanced US legal cases. Both projects used MiniMax-M3 with a 4,096-token output budget, temperature 0.1, a 300-second per-project/case timeout, and the identical locally generated vector-retrieval evidence.

| Metric | CodexLaw | Lawgent |
| --- | ---: | ---: |
| Completed calls | 60 / 60 | 60 / 60 |
| Correct answers | 44 / 60 (73.3%) | 40 / 60 (66.7%) |
| Valid citations | 48 / 60 (80.0%) | 49 / 60 (81.7%) |
| Correct answer **and** valid citation | **41 / 60 (68.3%)** | **35 / 60 (58.3%)** |
| Mean elapsed time | **2.810 s** | 16.249 s |
| Median elapsed time | **1.944 s** | 13.083 s |

The paired comparison has 14 CodexLaw-only successes, 8 Lawgent-only successes, 27 shared successes, and 11 shared failures. The two-sided exact McNemar p-value is 0.286279, so this 60-case batch is evidence of a practical advantage for CodexLaw but not a statistically conclusive difference at a 5% threshold.

## By source

| Source (20 cases each) | CodexLaw: correct + valid citation | Lawgent: correct + valid citation |
| --- | ---: | ---: |
| CaseHOLD | **14 / 20** | 8 / 20 |
| LegalBench / Abercrombie | **16 / 20** | 15 / 20 |
| LegalBench-RAG / CUAD | 11 / 20 | **12 / 20** |

## Retrieval and scope

The retrieval input came from the local 58,604-record embedding index, using the private DGX Spark `nvidia/Nemotron-3-Embed-1B-BF16` endpoint. Each case was augmented with up to two eCFR and two LegalBench-RAG vector hits, while preserving its original evidence. No Internet retrieval was used.

This report is deliberately labelled **offline vector-RAG**, not Neo4j graph-RAG. At the time of the run, Docker Desktop could not start because WSL/HCS could not mount its system VHD; the local Neo4j Bolt endpoint was unavailable. Therefore Neo4j did not participate in this batch. A graph-backed rerun remains required after Docker and WSL are healthy.

## Reproduction record

The augmented input was created with `scripts/prepare_offline_rag_cases.py`; raw input and output rows are ignored under `benchmark/results/`:

- input: `offline-rag-cases-20260903T084700Z.jsonl`
- rows: `project-ab-20260903T085033Z.jsonl`
- model: MiniMax-M3
- output budget: 4,096 tokens
- temperature: 0.1
- timeout: 300 seconds per project/case

The same evaluator and output contract were used for both projects. Project-specific workflow-node counts were excluded from the shared score.
