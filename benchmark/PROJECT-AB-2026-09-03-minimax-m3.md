# Lawgent versus CodexLaw project benchmark: MiniMax-M3

## Scope

This is a project-to-project closed-book comparison, not an internal prompt or model comparison.

- **Lawgent** runs its original `WorkflowExecutor`, planner, routing, and workflow code from `vendor/lawgent`.
- **CodexLaw** runs its Codex orchestration path.
- Both projects use the locally configured `mmx` CLI with `MiniMax-M3`, a 4,096-token output ceiling, temperature 0.1, and a 300-second per-project timeout.
- Both receive the same selected public case and only its supplied benchmark evidence. External retrieval is disabled for both.
- The common score is answer correctness plus validity of a supplied evidence ID. Project-specific workflow-node counts are excluded because they do not have the same meaning across the two systems.
- Lawgent's native user-facing Markdown is deterministically converted to the shared response schema only when it explicitly states an `Answer:`. The converter extracts that answer and only evidence IDs already present in the response; it never calls a model, chooses an answer, or reads the reference answer.

## Result

Run `20260903T041545Z` evaluated a deterministic, source-balanced 60-case batch: 20 LegalBench classification cases, 20 LegalBench-RAG CUAD evidence-span cases, and 20 CaseHOLD multiple-choice cases.

Before this run, the Lawgent adapter was corrected in three ways: its internal specialists now use the same MiniMax provider as its parent workflow; MiniMax child processes always use UTF-8 on Windows; and the shared evaluator extracts an explicitly labelled answer from Lawgent's natural-language terminal response without a second model call. These changes remove adapter failures and format bias without changing either project's orchestration.

| Project | Execution completed | Correct when completed | Valid citation when completed | Common success |
| --- | ---: | ---: | ---: | ---: |
| CodexLaw | 60 / 60 | 47 / 60 | 54 / 60 | 43 / 60 |
| Lawgent | 60 / 60 | 47 / 60 | 56 / 60 | 47 / 60 |

Both projects completed every row with zero provider errors and had the same answer correctness: 47 of 60. All 47 Lawgent answer-correct rows had valid citations; CodexLaw had four answer-correct rows with an invalid citation, which accounts for the four-point common-success difference. The source breakdown is below.

| Project | CaseHOLD | LegalBench | LegalBench-RAG | Mean wall time per case |
| --- | ---: | ---: | ---: | ---: |
| CodexLaw | 13 / 20 | 16 / 20 | 14 / 20 | 2.17 s |
| Lawgent | 15 / 20 | 19 / 20 | 13 / 20 | 9.14 s |

The raw machine-readable result is `benchmark/results/project-ab-20260903T041545Z.jsonl`; its summary is `benchmark/results/project-ab-20260903T041545Z.summary.json`. These paths are deliberately ignored because they contain provider outputs.

## Interpretation

In paired outcomes, CodexLaw alone succeeded on five cases, Lawgent alone succeeded on nine, and both had the same result on 46. With only 14 discordant pairs, the two-sided exact paired test is 0.424; this is not a statistically significant quality ranking. CodexLaw is nevertheless about 4.2 times faster in mean wall time on this host. Extend only with deterministic, source-balanced batches while preserving the model, output budget, closed-book evidence rule, answer-type output contract, common evaluator, and timeout policy. The earlier K3 report remains integration evidence for CodexLaw and must not be used to infer Lawgent-versus-CodexLaw performance.
