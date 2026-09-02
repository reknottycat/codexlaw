#!/usr/bin/env python3
"""Extract the public LegalBench-RAG archive with a Windows-safe path map."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


_INVALID = re.compile(r"[<>:\"/\\|?*]")
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def _safe_component(component: str) -> str:
    result: list[str] = []
    for char in component:
        if char == "%":
            result.append("%25")
        elif _INVALID.fullmatch(char):
            result.append(f"%{ord(char):02X}")
        else:
            result.append(char)
    safe = "".join(result).rstrip(" .")
    if not safe:
        safe = "%20"
    if safe.upper().split(".", 1)[0] in _RESERVED:
        safe = f"%5B{safe}%5D"
    return safe


def _safe_path(name: str) -> str:
    parts = [part for part in PurePosixPath(name).parts if part not in {"", "."}]
    return "/".join(_safe_component(part) for part in parts)


def extract(archive: Path, output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    mapping: list[dict[str, str]] = []
    extracted_files = 0
    with ZipFile(archive) as source:
        for info in source.infolist():
            if info.filename.endswith("/"):
                continue
            safe_name = _safe_path(info.filename)
            target = output / safe_name
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            mapping.append({"original_path": info.filename, "safe_path": safe_name})
            extracted_files += 1
    manifest = {
        "archive": str(archive),
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "files": extracted_files,
        "path_mappings": mapping,
    }
    (output / "path-map.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path("data/raw/legalbench-rag.zip"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/legalbench-rag"))
    args = parser.parse_args()
    manifest = extract(args.archive, args.output)
    print(f"extracted={manifest['files']}")
    print(f"path_map={args.output / 'path-map.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
