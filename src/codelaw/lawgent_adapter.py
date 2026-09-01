"""Architecture A: use Lawgent domain logic without replacing its legal semantics."""

from __future__ import annotations

import json
from collections.abc import Callable


class LawgentAdapter:
    def __init__(self, extractor: Callable[[str], str] | None = None):
        self.extractor = extractor or self._lawgent_extractor

    @staticmethod
    def _lawgent_extractor(clause: str) -> str:
        try:
            from legal_helper.tools.contract import extract_clauses
        except ImportError as exc:
            raise RuntimeError("Lawgent is not installed; run scripts/bootstrap_sources.sh then install vendor/lawgent") from exc
        return extract_clauses(text=clause)

    def contract_intake(self, clause: str) -> dict[str, object]:
        extracted = json.loads(self.extractor(clause))
        categories = sorted({category for row in extracted for category in row.get("categories", [])})
        return {"architecture": "lawgent", "clause": clause, "categories": categories, "raw": extracted}
