#!/usr/bin/env python3
"""Download official eCFR title XML and retain exact source provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", type=int, required=True)
    parser.add_argument("--issue-date", required=True, help="YYYY-MM-DD version date from eCFR")
    parser.add_argument("--out", type=Path, default=Path("data/sources/ecfr"))
    args = parser.parse_args()
    url = f"https://www.ecfr.gov/api/versioner/v1/full/{args.issue_date}/title-{args.title}.xml"
    payload = urlopen(url, timeout=60).read()
    args.out.mkdir(parents=True, exist_ok=True)
    xml_path = args.out / f"title-{args.title}-{args.issue_date}.xml"
    xml_path.write_bytes(payload)
    (xml_path.with_suffix(".manifest.json")).write_text(json.dumps({"source_url": url, "issue_date": args.issue_date, "sha256": hashlib.sha256(payload).hexdigest()}, indent=2), encoding="utf-8")
    print(xml_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
