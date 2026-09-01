"""Independent citation validation; never delegate this decision to the agent."""

from __future__ import annotations

from dataclasses import dataclass

from .evidence import EvidenceLedger


@dataclass(frozen=True)
class CitationResult:
    citation_valid: bool
    source_exists: bool
    jurisdiction_valid: bool
    effective_date_valid: bool
    supports_claim: bool
    confidence: float


class CitationVerifier:
    def verify(self, ledger: EvidenceLedger, evidence_id: str, *, jurisdiction: str, effective_on: str, claim_terms: tuple[str, ...]) -> CitationResult:
        source_exists = evidence_id in ledger.by_id
        if not source_exists:
            return CitationResult(False, False, False, False, False, 0.0)
        evidence = ledger.by_id[evidence_id]
        jurisdiction_valid = evidence.jurisdiction == jurisdiction
        effective_date_valid = evidence.effective_on is None or evidence.effective_on <= effective_on
        supports_claim = all(term.lower() in evidence.text.lower() for term in claim_terms)
        valid = source_exists and jurisdiction_valid and effective_date_valid and supports_claim
        return CitationResult(valid, source_exists, jurisdiction_valid, effective_date_valid, supports_claim, 0.95 if valid else 0.0)
