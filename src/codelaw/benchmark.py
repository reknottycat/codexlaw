"""Reusable A/B benchmark primitives for public legal evaluation data."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .citations import CitationVerifier
from .codex_legal_adapter import AssistantTurn, CodexLegalAdapter
from .evidence import Evidence, EvidenceLedger
from .lawgent_adapter import LawgentAdapter
from .workflow import REQUIRED_NODES, LegalWorkflow, WorkflowState


@dataclass(frozen=True)
class BenchmarkDecision:
    answer: str
    citation_ids: list[str]
    confidence: float | None
    raw_response: str
    parse_error: str | None = None


def iter_cases(path: Path, *, sources: set[str] | None = None) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            case = json.loads(line)
            if sources and case.get("source") not in sources:
                continue
            yield case


def select_cases(path: Path, *, limit: int, sources: list[str] | None = None) -> list[dict[str, Any]]:
    """Select a deterministic, source-balanced sample without loading 140k rows."""
    if limit < 1:
        raise ValueError("limit must be positive")
    if not sources:
        return list(_take(iter_cases(path), limit))
    ordered_sources = list(dict.fromkeys(sources))
    buckets = {source: [] for source in ordered_sources}
    for case in iter_cases(path, sources=set(ordered_sources)):
        source = case.get("source")
        if source in buckets and len(buckets[source]) < limit:
            buckets[source].append(case)
        if all(len(rows) >= limit for rows in buckets.values()):
            break
    selected: list[dict[str, Any]] = []
    while len(selected) < limit:
        added = False
        for source in ordered_sources:
            bucket = buckets[source]
            if bucket:
                selected.append(bucket.pop(0))
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
    return selected


def _take(rows: Iterable[dict[str, Any]], limit: int) -> Iterator[dict[str, Any]]:
    for index, row in enumerate(rows):
        if index == limit:
            return
        yield row


def _normalise(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _choice_index(answer: str) -> int | None:
    value = _normalise(answer).strip("`*.,:;()[]")
    match = re.search(r"\b(?:option\s*)?([a-e])\b", value)
    if match:
        return ord(match.group(1)) - ord("a")
    match = re.search(r"(?:^|\s)([0-4])(?:$|\s)", value)
    return int(match.group(1)) if match else None


def answer_matches(*, answer: str, expected: str, answer_type: str) -> bool:
    if answer_type == "multiple_choice":
        actual_index = _choice_index(answer)
        try:
            expected_index = int(float(expected))
        except (TypeError, ValueError):
            expected_index = _choice_index(expected)
        return actual_index is not None and actual_index == expected_index
    actual = _normalise(answer)
    target = _normalise(expected)
    if answer_type != "evidence_span":
        return actual == target
    if not actual or not target:
        return False
    if actual.strip(".!? ") in {"yes", "no"}:
        return False
    if actual == target:
        return True
    actual_tokens = set(re.findall(r"\w+", actual))
    target_tokens = set(re.findall(r"\w+", target))
    if not actual_tokens or not target_tokens:
        return False
    overlap = len(actual_tokens & target_tokens)
    if overlap < 4:
        return False
    if actual in target or target in actual:
        return True
    # Benchmarks often store a full contractual sentence as the reference while
    # the task requests the shortest supported span. Require a substantial,
    # non-trivial portion of the submitted answer to be grounded in that span.
    return overlap / len(actual_tokens) >= 0.8


def parse_decision(raw_response: str) -> BenchmarkDecision:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", raw_response):
        try:
            payload, _ = decoder.raw_decode(raw_response[match.start():])
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        answer = payload.get("answer")
        if answer is None:
            return BenchmarkDecision("", [], None, raw_response, "JSON response did not contain answer")
        citation_ids = payload.get("citation_ids", [])
        if isinstance(citation_ids, str):
            citation_ids = [citation_ids]
        if not isinstance(citation_ids, list):
            citation_ids = []
        confidence = payload.get("confidence")
        try:
            confidence = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence = None
        return BenchmarkDecision(str(answer), [str(item) for item in citation_ids], confidence, raw_response)
    return BenchmarkDecision("", [], None, raw_response, "Response did not contain a JSON object")


def build_prompt(case: dict[str, Any], *, architecture: str, categories: list[str], max_evidence_chars: int) -> str:
    answer_type = case.get("answer_type", "classification")
    if answer_type == "multiple_choice":
        output_rule = "For multiple choice, answer with the zero-based option index or its letter."
    elif answer_type == "evidence_span":
        output_rule = "For evidence-span questions, answer with the shortest supported text or a faithful concise answer."
    else:
        output_rule = "For classification questions, answer with exactly one label from the task's answer space."
    evidence_blocks: list[str] = []
    remaining = max_evidence_chars
    for evidence in case.get("evidence", []):
        text = str(evidence.get("text", ""))
        if remaining <= 0:
            break
        excerpt = text[:remaining]
        remaining -= len(excerpt)
        evidence_blocks.append(
            f"[{evidence.get('evidence_id')}] source={evidence.get('source_id')} "
            f"jurisdiction={evidence.get('jurisdiction')} effective_on={evidence.get('effective_on')}\n{excerpt}"
        )
    category_hint = ", ".join(categories) if categories else "none"
    return (
        "You are evaluating one public legal benchmark item. Use only the supplied evidence. "
        "Do not invent authorities or facts. Return one JSON object and no other visible text "
        "with exactly these keys: answer, citation_ids, confidence. citation_ids must contain "
        "only supplied evidence IDs. confidence must be a number from 0 to 1. "
        f"Architecture={architecture}. {output_rule}\n\n"
        f"Task={case.get('task')}\nQuestion:\n{case.get('prompt', '').strip()}\n\n"
        f"Lawgent clause-category hints (not an answer): {category_hint}\n\n"
        "Evidence:\n" + "\n\n".join(evidence_blocks)
    )


def _claim_terms(case: dict[str, Any]) -> tuple[str, ...]:
    if case.get("answer_type") != "evidence_span":
        return ()
    return tuple(re.findall(r"[A-Za-z0-9]{4,}", str(case.get("expected_answer", "")))[:8])


def _ledger(case: dict[str, Any]) -> EvidenceLedger:
    ledger = EvidenceLedger()
    for row in case.get("evidence", []):
        ledger.add(Evidence(
            evidence_id=str(row["evidence_id"]),
            source_id=str(row.get("source_id", "")),
            text=str(row["text"]),
            jurisdiction=str(row.get("jurisdiction", case.get("jurisdiction", "US"))),
            effective_on=row.get("effective_on"),
        ))
    return ledger


def run_case(
    case: dict[str, Any],
    *,
    architecture: str,
    ask: Callable[[str], str],
    lawgent: LawgentAdapter | None = None,
    max_evidence_chars: int = 12000,
) -> dict[str, Any]:
    if architecture not in {"A", "B"}:
        raise ValueError("architecture must be A or B")
    started = time.monotonic()
    categories: list[str] = []
    raw_response = ""
    error: str | None = None
    decision = BenchmarkDecision("", [], None, "")
    state = WorkflowState()
    citation_valid = False
    workflow_compliant = False
    try:
        ledger = _ledger(case)
        if architecture == "A":
            intake = (lawgent or LawgentAdapter()).contract_intake(
                "\n\n".join(str(row.get("text", "")) for row in case.get("evidence", []))
            )
            categories = [str(item) for item in intake.get("categories", [])]
        state.complete("CONTRACT_INTAKE")
        state.complete("JURISDICTION")
        state.complete("EFFECTIVE_DATE")
        state.complete("EXCEPTION_CHECK")
        raw_response = ask(build_prompt(case, architecture=architecture, categories=categories, max_evidence_chars=max_evidence_chars))
        decision = parse_decision(raw_response)
        if decision.parse_error:
            error = decision.parse_error
        state.citation_id = decision.citation_ids[0] if decision.citation_ids else None
        effective_on = str(case.get("effective_on") or "9999-12-31")
        verifier = CitationVerifier()
        verification = verifier.verify(
            ledger,
            state.citation_id or "",
            jurisdiction=str(case.get("jurisdiction") or "US"),
            effective_on=effective_on,
            claim_terms=_claim_terms(case),
        )
        citation_valid = verification.citation_valid
        if citation_valid:
            state.complete("CITATION_VERIFICATION")
        workflow = LegalWorkflow(ledger, verifier)
        if architecture == "B":
            adapter = CodexLegalAdapter(lambda _: AssistantTurn(raw_response), workflow)
            adapter.analyze(
                "final review",
                state,
                jurisdiction=str(case.get("jurisdiction") or "US"),
                effective_on=effective_on,
                claim_terms=_claim_terms(case),
            )
        else:
            workflow.final_review(
                state,
                jurisdiction=str(case.get("jurisdiction") or "US"),
                effective_on=effective_on,
                claim_terms=_claim_terms(case),
            )
        workflow_compliant = set(REQUIRED_NODES).issubset(state.completed)
    except Exception as exc:  # benchmark rows must remain inspectable after one bad row
        error = str(exc)
    answer_correct = answer_matches(
        answer=decision.answer,
        expected=str(case.get("expected_answer", "")),
        answer_type=str(case.get("answer_type", "classification")),
    )
    return {
        "case_id": case.get("case_id"),
        "source": case.get("source"),
        "task": case.get("task"),
        "answer_type": case.get("answer_type"),
        "architecture": architecture,
        "answer": decision.answer,
        "expected_answer": case.get("expected_answer"),
        "citation_ids": decision.citation_ids,
        "confidence": decision.confidence,
        "answer_correct": answer_correct,
        "citation_valid": citation_valid,
        "workflow_compliant": workflow_compliant,
        "success": answer_correct and citation_valid and workflow_compliant,
        "completed_nodes": state.completed,
        "lawgent_categories": categories,
        "model_response": raw_response,
        "error": error,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def run_ab(
    cases: list[dict[str, Any]],
    *,
    ask: Callable[[str], str],
    interval_seconds: float,
    lawgent: LawgentAdapter | None = None,
    sleep: Callable[[float], None] = time.sleep,
    max_evidence_chars: int = 12000,
    on_row: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    first_request = True
    for case in cases:
        for architecture in ("A", "B"):
            if not first_request:
                sleep(interval_seconds)
            first_request = False
            row = run_case(
                case,
                architecture=architecture,
                ask=ask,
                lawgent=lawgent,
                max_evidence_chars=max_evidence_chars,
            )
            rows.append(row)
            if on_row is not None:
                on_row(row)
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["architecture"]), []).append(row)
    summary: dict[str, Any] = {"total_rows": len(rows), "architectures": {}}
    for architecture, group in groups.items():
        total = len(group)
        summary["architectures"][architecture] = {
            "rows": total,
            "answer_accuracy": sum(bool(row["answer_correct"]) for row in group) / max(1, total),
            "citation_validity": sum(bool(row["citation_valid"]) for row in group) / max(1, total),
            "workflow_compliance": sum(bool(row["workflow_compliant"]) for row in group) / max(1, total),
            "success_rate": sum(bool(row["success"]) for row in group) / max(1, total),
            "errors": sum(bool(row["error"]) for row in group),
        }
    return summary
