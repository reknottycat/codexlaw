# Lawgent versus CodexLaw project benchmark: MiniMax-M3

## Scope

This is a project-to-project closed-book comparison, not an internal prompt or model comparison.

- **Lawgent** runs its original `WorkflowExecutor`, planner, routing, and workflow code from `vendor/lawgent`.
- **CodexLaw** runs its Codex orchestration path.
- Both projects use the locally configured `mmx` CLI with `MiniMax-M3`, a 4,096-token output ceiling, temperature 0.1, and a 300-second per-project timeout.
- Both receive the same selected public case and only its supplied benchmark evidence. External retrieval is disabled for both.
- The common score is answer correctness plus validity of a supplied evidence ID. Project-specific workflow-node counts are excluded because they do not have the same meaning across the two systems.

## Result

Run `20260903T034220Z` evaluated one deterministic case from each public source.

| Project | Execution completed | Correct when completed | Valid citation when completed | Common success |
| --- | ---: | ---: | ---: | ---: |
| CodexLaw | 3 / 3 | 2 / 3 | 3 / 3 | 2 / 3 |
| Lawgent | 3 / 3 | 2 / 3 | 3 / 3 | 2 / 3 |

The completed LegalBench classification and CaseHOLD multiple-choice rows passed for both projects. On the LegalBench-RAG CUAD evidence-span row, both projects returned valid citations but omitted enough reference conditions to fail the shared strict answer matcher. This sample therefore shows no quality winner.

The raw machine-readable result is `benchmark/results/project-ab-20260903T034220Z.jsonl`; its summary is `benchmark/results/project-ab-20260903T034220Z.summary.json`. These paths are deliberately ignored because they contain provider outputs.

## Interpretation

This is a valid first cross-source baseline, not a statistically significant ranking. Extend it with deterministic, source-balanced batches only after preserving the same model, output budget, closed-book evidence rule, common evaluator, and timeout policy. The earlier K3 report remains integration evidence for CodexLaw and must not be used to infer Lawgent-versus-CodexLaw performance.
