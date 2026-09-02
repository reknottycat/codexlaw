"""Architecture A: use Lawgent domain logic without replacing its legal semantics."""

from __future__ import annotations

import json
from collections.abc import Callable
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


class LawgentAdapter:
    def __init__(self, extractor: Callable[[str], str] | None = None):
        self.extractor = extractor or self._lawgent_extractor

    @staticmethod
    def _lawgent_extractor(clause: str) -> str:
        try:
            from legal_helper.tools.contract import extract_clauses
        except ImportError:
            contract_path = Path(__file__).resolve().parents[2] / "vendor" / "lawgent" / "legal_helper" / "tools" / "contract.py"
            if not contract_path.exists():
                raise RuntimeError("Lawgent is not installed; run scripts/bootstrap_sources.sh then install vendor/lawgent")
            spec = spec_from_file_location("codelaw._lawgent_contract", contract_path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Lawgent contract extractor could not be loaded from {contract_path}")
            module = module_from_spec(spec)
            spec.loader.exec_module(module)
            extract_clauses = module.extract_clauses
        return extract_clauses(text=clause)

    def contract_intake(self, clause: str) -> dict[str, object]:
        extracted = json.loads(self.extractor(clause))
        categories = sorted({category for row in extracted for category in row.get("categories", [])})
        return {"architecture": "lawgent", "clause": clause, "categories": categories, "raw": extracted}
