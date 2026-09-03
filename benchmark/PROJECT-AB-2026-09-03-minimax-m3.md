# Lawgent versus CodexLaw project benchmark: MiniMax-M3

## Scope

This is a project-to-project closed-book comparison, not an internal prompt or model comparison.

- **Lawgent** runs its original `WorkflowExecutor`, planner, routing, and workflow code from `vendor/lawgent`.
- **CodexLaw** runs its Codex orchestration path.
- Both projects use the locally configured `mmx` CLI with `MiniMax-M3`, a 4,096-token output ceiling, temperature 0.1, and a 300-second per-project timeout.
- Both receive the same selected public case and only its supplied benchmark evidence. External retrieval is disabled for both.
- The common score is answer correctness plus validity of a supplied evidence ID. Project-specific workflow-node counts are excluded because they do not have the same meaning across the two systems.

## Result

Run `20260903T035031Z` evaluated a deterministic, source-balanced 12-case batch: four LegalBench classification cases, four LegalBench-RAG CUAD evidence-span cases, and four CaseHOLD multiple-choice cases.

Before this run, the Lawgent input adapter was corrected to include the same answer-type output contract as CodexLaw: a single classification label, only the option index or letter for multiple choice, and a concise supported answer for evidence spans. This prevents the evaluator from treating an otherwise correct but explanatory classification response as a different answer.

| Project | Execution completed | Correct when completed | Valid citation when completed | Common success |
| --- | ---: | ---: | ---: | ---: |
| CodexLaw | 12 / 12 | 10 / 12 | 12 / 12 | 10 / 12 |
| Lawgent | 12 / 12 | 11 / 12 | 12 / 12 | 11 / 12 |

Both projects completed every row and supplied valid evidence identifiers for every completed answer. They both passed all eight LegalBench and LegalBench-RAG rows. The only shared wrong answer was `casehold:all:2`; CodexLaw also missed `casehold:all:1`, while Lawgent answered that case correctly. The source breakdown is below.

| Project | CaseHOLD | LegalBench | LegalBench-RAG | Mean wall time per case |
| --- | ---: | ---: | ---: | ---: |
| CodexLaw | 2 / 4 | 4 / 4 | 4 / 4 | 2.02 s |
| Lawgent | 3 / 4 | 4 / 4 | 4 / 4 | 9.80 s |

The raw machine-readable result is `benchmark/results/project-ab-20260903T035031Z.jsonl`; its summary is `benchmark/results/project-ab-20260903T035031Z.summary.json`. These paths are deliberately ignored because they contain provider outputs.

## Interpretation

This single 12-case run favors Lawgent by one CaseHOLD answer, while CodexLaw is about five times faster in wall time on this host. It is not a statistically significant quality ranking: MiniMax responses are nondeterministic and the sample is small. Extend only with deterministic, source-balanced batches while preserving the model, output budget, closed-book evidence rule, answer-type output contract, common evaluator, and timeout policy. The earlier K3 report remains integration evidence for CodexLaw and must not be used to infer Lawgent-versus-CodexLaw performance.
