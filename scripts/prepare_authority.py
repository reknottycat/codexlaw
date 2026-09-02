#!/usr/bin/env python3
"""Normalize downloaded official eCFR XML into auditable authority JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from codelaw.ecfr import parse_title_xml


_XML_NAME = re.compile(r"title-(?P<title>\d+)-(?P<date>\d{4}-\d{2}-\d{2})\.xml$")


def _source_metadata(xml_path: Path) -> tuple[int, str, str]:
    manifest_path = xml_path.with_suffix(".manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        title_match = re.search(r"title-(\d+)", xml_path.name)
        if not title_match:
            raise ValueError(f"Cannot infer eCFR title from {xml_path}")
        return int(title_match.group(1)), manifest["source_url"], manifest["issue_date"]
    match = _XML_NAME.fullmatch(xml_path.name)
    if not match:
        raise ValueError(f"Expected title-N-YYYY-MM-DD.xml: {xml_path}")
    title = int(match.group("title"))
    issue_date = match.group("date")
    source_url = f"https://www.ecfr.gov/api/versioner/v1/full/{issue_date}/title-{title}.xml"
    return title, source_url, issue_date


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("xml_paths", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/authority"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "authority.jsonl"
    counts: dict[str, int] = {}
    sources: list[dict[str, object]] = []
    total = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for xml_path in args.xml_paths:
            payload = xml_path.read_bytes()
            title, source_url, issue_date = _source_metadata(xml_path)
            records = parse_title_xml(payload, title=title, source_url=source_url, issue_date=issue_date)
            for record in records:
                output.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            key = f"title-{title}:{issue_date}"
            counts[key] = len(records)
            total += len(records)
            sources.append({
                "title": title,
                "issue_date": issue_date,
                "source_url": source_url,
                "local_path": str(xml_path),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "records": len(records),
            })
    manifest = {
        "schema_version": "1.0.0",
        "records_path": str(output_path),
        "total": total,
        "counts": counts,
        "sources": sources,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
