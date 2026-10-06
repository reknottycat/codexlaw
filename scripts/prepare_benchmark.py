#!/usr/bin/env python3
"""Normalize public legal benchmarks into one auditable JSONL stream."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterator


SCHEMA_VERSION = "1.0.0"
RAG_BENCHMARKS = ("cuad", "contractnli", "maud", "privacy_qa")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jurisdiction(task: str) -> str:
    task_lower = task.lower()
    if "canada" in task_lower:
        return "CA"
    if "europe" in task_lower or "eu_" in task_lower:
        return "EU"
    return "US"


def _render_instruction(template: str, row: dict[str, str]) -> str:
    return re.sub(r"\{\{([^}]+)\}\}", lambda match: row.get(match.group(1), ""), template).strip()


def _evidence_text(row: dict[str, str]) -> str:
    for key in ("text", "contract", "policy", "claim", "Paragraph", "paragraph"):
        value = row.get(key, "").strip()
        if value:
            return value
    return ""


def _case(
    *,
    case_id: str,
    source: str,
    task: str,
    prompt: str,
    expected_answer: str,
    jurisdiction: str,
    evidence: list[dict[str, Any]],
    answer_type: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "source": source,
        "task": task,
        "prompt": prompt,
        "expected_answer": expected_answer,
        "answer_type": answer_type,
        "jurisdiction": jurisdiction,
        "effective_on": None,
        "evidence": evidence,
        "metadata": metadata or {},
    }


def _iter_legalbench(root: Path) -> Iterator[dict[str, Any]]:
    metadata = json.loads((root / "task_metadata.json").read_text(encoding="utf-8"))
    for task_dir in sorted(path for path in (root / "data").iterdir() if path.is_dir()):
        test_path = task_dir / "test.tsv"
        if not test_path.exists() or task_dir.name not in metadata:
            continue
        instruction = metadata[task_dir.name].get("instruction", "")
        answer_space = metadata[task_dir.name].get("answer_space", [])
        with test_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                index = row.get("index", "0")
                evidence_text = _evidence_text(row)
                if not evidence_text:
                    evidence_text = _render_instruction(instruction, row)
                case_id = f"legalbench:{task_dir.name}:{index}"
                yield _case(
                    case_id=case_id,
                    source="legalbench",
                    task=task_dir.name,
                    prompt=_render_instruction(instruction, row),
                    expected_answer=row.get("answer", "").strip(),
                    jurisdiction=_jurisdiction(task_dir.name),
                    evidence=[{
                        "evidence_id": f"{case_id}:evidence",
                        "source_id": case_id,
                        "text": evidence_text,
                        "jurisdiction": _jurisdiction(task_dir.name),
                        "effective_on": None,
                    }],
                    answer_type="classification",
                    metadata={
                        "eval_method": metadata[task_dir.name].get("eval_method"),
                        "answer_space": answer_space,
                        "row_fields": sorted(row),
                        "source_path": str(test_path),
                    },
                )


def _rag_path_map(root: Path) -> dict[str, str]:
    payload = json.loads((root / "path-map.json").read_text(encoding="utf-8"))
    result: dict[str, str] = {}
    for row in payload["path_mappings"]:
        original_path = row["original_path"]
        result[original_path] = row["safe_path"]
        result[unicodedata.normalize("NFC", original_path)] = row["safe_path"]
    return result


def _iter_rag(root: Path) -> Iterator[dict[str, Any]]:
    path_map = _rag_path_map(root)
    file_cache: dict[str, str] = {}
    for benchmark_name in RAG_BENCHMARKS:
        benchmark_path = root / "benchmarks" / f"{benchmark_name}.json"
        if not benchmark_path.exists():
            continue
        payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
        for index, test in enumerate(payload.get("tests", [])):
            case_id = f"legalbench-rag:{benchmark_name}:{index}"
            evidence: list[dict[str, Any]] = []
            gold_paths: list[dict[str, Any]] = []
            for snippet_index, snippet in enumerate(test.get("snippets", [])):
                original_path = str(snippet["file_path"])
                archive_path = f"corpus/{original_path}"
                safe_path = path_map.get(archive_path) or path_map.get(
                    unicodedata.normalize("NFC", archive_path)
                )
                if safe_path is None:
                    raise FileNotFoundError(f"No safe path mapping for LegalBench-RAG file: {archive_path}")
                if safe_path not in file_cache:
                    file_cache[safe_path] = (root / safe_path).read_text(encoding="utf-8", errors="replace")
                document = file_cache[safe_path]
                span = snippet.get("span") or [0, 0]
                start, end = int(span[0]), int(span[1])
                text = document[start:end].strip()
                if not text:
                    text = str(snippet.get("answer", "")).strip()
                evidence_id = f"{case_id}:snippet:{snippet_index}"
                evidence.append({
                    "evidence_id": evidence_id,
                    "source_id": f"legalbench-rag:{benchmark_name}:{original_path}",
                    "text": text,
                    "jurisdiction": "US",
                    "effective_on": None,
                })
                gold_paths.append({"evidence_id": evidence_id, "file_path": original_path, "span": [start, end]})
            expected = "\n".join(str(item.get("answer", "")).strip() for item in test.get("snippets", []) if item.get("answer"))
            yield _case(
                case_id=case_id,
                source="legalbench-rag",
                task=benchmark_name,
                prompt=str(test.get("query", "")).strip(),
                expected_answer=expected,
                jurisdiction="US",
                evidence=evidence,
                answer_type="evidence_span",
                metadata={
                    "gold_evidence": gold_paths,
                    "benchmark_path": str(benchmark_path),
                    "corpus_root": str(root / "corpus"),
                },
            )


def _iter_casehold(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row_number, row in enumerate(reader):
            if len(row) < 13:
                continue
            try:
                label = int(float(row[-1]))
            except ValueError:
                continue
            context = row[1].strip()
            choices = [choice.strip() for choice in row[2:7]]
            if not context or label not in range(5) or any(not choice for choice in choices):
                continue
            options = "\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices))
            prompt = (
                "Select the one legally correct holding that best completes the context. "
                "Return the option letter or index and cite the supplied case context.\n\n"
                f"Context:\n{context}\n\nCandidates:\n{options}"
            )
            case_id = f"casehold:all:{row_number}"
            yield _case(
                case_id=case_id,
                source="casehold",
                task="casehold",
                prompt=prompt,
                expected_answer=str(label),
                jurisdiction="US",
                evidence=[{
                    "evidence_id": f"{case_id}:evidence",
                    "source_id": case_id,
                    "text": context,
                    "jurisdiction": "US",
                    "effective_on": None,
                }],
                answer_type="multiple_choice",
                metadata={"choices": choices, "label_index": label, "source_path": str(path)},
            )


def _source_entry(source_id: str, url: str, path: Path, license_name: str) -> dict[str, Any]:
    return {
        "id": source_id,
        "url": url,
        "local_path": str(path),
        "license": license_name,
        "sha256": _sha256(path) if path.is_file() else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legalbench", type=Path, default=Path("data/raw/legalbench-hf"))
    parser.add_argument("--casehold", type=Path, default=Path("data/raw/casehold-hf/data/all/train.csv"))
    parser.add_argument("--rag", type=Path, default=Path("data/processed/legalbench-rag"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/benchmark"))
    parser.add_argument("--skip-casehold", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "cases.jsonl"
    counts: dict[str, int] = {}
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        if args.rag.exists():
            for case in _iter_rag(args.rag):
                output.write(json.dumps(case, ensure_ascii=False) + "\n")
                counts[case["source"]] = counts.get(case["source"], 0) + 1
        if args.legalbench.exists():
            for case in _iter_legalbench(args.legalbench):
                output.write(json.dumps(case, ensure_ascii=False) + "\n")
                counts[case["source"]] = counts.get(case["source"], 0) + 1
        if not args.skip_casehold and args.casehold.exists():
            for case in _iter_casehold(args.casehold):
                output.write(json.dumps(case, ensure_ascii=False) + "\n")
                counts[case["source"]] = counts.get(case["source"], 0) + 1
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "cases_path": str(output_path),
        "counts": counts,
        "total": sum(counts.values()),
        "sources": [
            _source_entry("legalbench", "https://huggingface.co/datasets/nguha/legalbench", args.legalbench / "README.md", "CC BY 4.0 (dataset card; task-specific terms apply)"),
            _source_entry("casehold", "https://huggingface.co/datasets/casehold/casehold", args.casehold, "CaseHOLD repository/dataset terms"),
            _source_entry("legalbench-rag", "https://github.com/zeroentropy-cc/legalbenchrag", args.rag / "path-map.json", "Upstream dataset terms; review source dataset terms"),
        ],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
