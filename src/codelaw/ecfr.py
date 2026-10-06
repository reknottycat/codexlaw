"""Parse official eCFR title XML into source-preserving authority records."""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass


@dataclass(frozen=True)
class AuthorityRecord:
    authority_id: str
    citation: str
    text: str
    source_url: str
    issue_date: str
    sha256: str


def parse_title_xml(xml_bytes: bytes, *, title: int, source_url: str, issue_date: str) -> list[AuthorityRecord]:
    digest = hashlib.sha256(xml_bytes).hexdigest()
    root = ET.fromstring(xml_bytes)
    records: list[AuthorityRecord] = []
    for section in root.findall(".//DIV8[@TYPE='SECTION']"):
        number = (section.findtext("SECTNO") or section.get("N") or "").strip()
        subject = (section.findtext("SUBJECT") or section.findtext("HEAD") or "").strip()
        body = " ".join(" ".join(section.itertext()).split())
        if number and body:
            citation_number = number if number.startswith("§") else f"§{number}"
            citation = f"{title} CFR {citation_number}"
            records.append(AuthorityRecord(f"ecfr:{title}:{number}", citation, body or subject, source_url, issue_date, digest))
    return records
