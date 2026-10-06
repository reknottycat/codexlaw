"""Shared input contract and provenance for project comparisons."""
from __future__ import annotations

import copy
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


def json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()


def file_hash(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def prepare_case(case: dict[str, Any], max_evidence_chars: int = 0) -> dict[str, Any]:
    """Apply a single shared budget before either harness sees the evidence.

    Zero means all supplied evidence. The original evaluator still uses the
    original case, so omitted evidence never quietly changes the answer key.
    """
    if max_evidence_chars < 0:
        raise ValueError('max_evidence_chars must be nonnegative')
    prepared = copy.deepcopy(case)
    prepared['evidence'] = []
    remaining = max_evidence_chars
    for item in case.get('evidence', []):
        if max_evidence_chars and remaining <= 0:
            break
        evidence = dict(item)
        text = str(evidence['text']).encode('utf-8', errors='replace').decode('utf-8')
        evidence['text'] = text[:remaining] if max_evidence_chars else text
        remaining -= len(evidence['text'])
        prepared['evidence'].append(evidence)
    return prepared


def shared_task(case: dict[str, Any]) -> str:
    """Identical user task for both harnesses; no reference answer is sent."""
    answer_type = case.get('answer_type', 'classification')
    rule = {
        'multiple_choice': 'For multiple choice, answer with only the zero-based option index or its letter.',
        'evidence_span': 'For evidence spans, answer with the shortest supported text or a faithful concise answer. Return the relevant clause, not only Yes or No.',
    }.get(answer_type, "For classification, answer with exactly one label from the task's answer space.")
    evidence = '\n\n'.join(
        f"[{item['evidence_id']}] source={item.get('source_id', '')} "
        f"jurisdiction={item.get('jurisdiction', 'US')} effective_on={item.get('effective_on')}\n{item['text']}"
        for item in case.get('evidence', [])
    )
    return (
        'This is a closed-book legal benchmark. Use only the supplied evidence; do not use '
        'external tools, sources, or unstated facts. Return exactly one JSON object and no '
        'other text with keys answer, citation_ids, confidence. citation_ids must contain only '
        f'the supplied bracketed evidence IDs, copied exactly. {rule}\n\n'
        f"Task: {case.get('task')}\nQuestion: {case.get('prompt')}\n\nEvidence:\n{evidence}"
    ).encode('utf-8', errors='replace').decode('utf-8')


def normalise_response(case: dict[str, Any], response: str) -> str:
    """Same explicit Markdown extraction on A and B; never infer an answer."""
    from .benchmark import parse_decision
    if parse_decision(response).answer:
        return response
    match = re.search(r'(?im)^\s*(?:[-*]\s*)?(?:\*{1,2})?(?:answer|答案)(?:\*{1,2})?\s*[:：]\s*(.+?)\s*$', response)
    if not match:
        return response
    answer = match.group(1).strip().strip('`* ')
    if case.get('answer_type') == 'multiple_choice':
        choice = re.search(r'\b([a-eA-E]|[0-4])\b', answer)
        if not choice:
            return response
        answer = choice.group(1).upper()
    elif case.get('answer_type') == 'classification':
        answer = answer.split()[0].strip('.,:;`* ')
    citations = [str(e['evidence_id']) for e in case.get('evidence', []) if str(e['evidence_id']) in response]
    return json.dumps(dict(answer=answer, citation_ids=citations, confidence=None), ensure_ascii=False)


def source_snapshot(root: Path) -> dict[str, Any]:
    """Record exact implementation hashes, including uncommitted edits."""
    def git(directory: Path, *args: str) -> str | None:
        result = subprocess.run(['git', '-C', str(directory), *args], capture_output=True, text=True, encoding='utf-8', errors='replace', check=False)
        return result.stdout.strip() if not result.returncode else None
    paths = sorted([*(root / 'src/codelaw').glob('*.py'), *(root / 'scripts').glob('*.py')])
    hashes = {p.relative_to(root).as_posix(): file_hash(p) for p in paths}
    return dict(git_commit=git(root, 'rev-parse', 'HEAD'), implementation_sha256=json_hash(hashes),
                files=hashes, upstream_revisions={name: git(root / 'vendor' / name, 'rev-parse', 'HEAD') for name in ['lawgent', 'codex']})
