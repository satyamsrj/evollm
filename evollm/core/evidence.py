import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class EvidenceClass(Enum):
    EXPLICIT_CORRECTION = "CORR"
    VERIFIED_FACT       = "KNOW"
    PREFERENCE          = "PREF"
    HYPOTHETICAL        = "HYPO"
    QUERY               = "QURY"


@dataclass
class EvidenceRecord:
    text:                str
    evidence_class:      EvidenceClass
    confidence:          float
    provenance_verified: bool
    source_reference:    Optional[str]
    accept_for_learning: bool
    rejection_reason:    Optional[str]
    loss_target:         Optional[str]


class EvidenceQualifier:
    _QUERY_PATTERNS = [
        r"^(what|who|where|when|why|how|can|could|would|will|is|are|do|does|did)\b",
        r"\?\s*$",
    ]
    _HYPOTHETICAL_PATTERNS = [
        r"\b(assume|suppose|imagine|pretend|hypothetically|let's say|what if|say that)\b",
        r"\b(for the sake of argument|just suppose|as if)\b",
    ]
    _CORRECTION_PATTERNS = [
        r"\b(no[,.]?|wrong[,.]?|incorrect[,.]?|actually[,.]?|not true[,.]?|correction[,:]?|that('s| is) (wrong|incorrect|not right))\b",
        r"\bit (should be|is actually|was actually)\b",
    ]
    _PREFERENCE_PATTERNS = [
        r"\bi (prefer|like|want|would rather|always wanted)\b",
        r"\b(shorter|more concise|more detailed|simpler|formal|informal) (answers?|responses?|replies?)\b",
    ]
    _ASSERTION_PATTERNS = [
        r"\b(is|are|was|were|has|have|uses?|runs? on|named|located in|updated to|changed to|now|currently|specif(ies|y)|set to|equal(s| to)|defined as|configured as)\b",
    ]

    def qualify(self, text: str, source_reference: Optional[str] = None) -> EvidenceRecord:
        t = text.strip().lower()

        if self._matches_any(t, self._QUERY_PATTERNS):
            return self._reject(text, EvidenceClass.QUERY, "QUERY_TURN")

        if self._matches_any(t, self._HYPOTHETICAL_PATTERNS):
            return self._reject(text, EvidenceClass.HYPOTHETICAL, "HYPOTHETICAL_STATEMENT")

        if self._matches_any(t, self._CORRECTION_PATTERNS):
            return EvidenceRecord(
                text=text,
                evidence_class=EvidenceClass.EXPLICIT_CORRECTION,
                confidence=0.9,
                provenance_verified=source_reference is not None,
                source_reference=source_reference,
                accept_for_learning=True,
                rejection_reason=None,
                loss_target="L_CORR",
            )

        if self._matches_any(t, self._PREFERENCE_PATTERNS):
            return EvidenceRecord(
                text=text,
                evidence_class=EvidenceClass.PREFERENCE,
                confidence=0.75,
                provenance_verified=True,
                source_reference=None,
                accept_for_learning=False,
                rejection_reason="L_PREF_DISABLED",
                loss_target=None,
            )

        if self._matches_any(t, self._ASSERTION_PATTERNS):
            if source_reference is None:
                return self._reject(text, EvidenceClass.VERIFIED_FACT, "SOURCE_UNAVAILABLE")
            return EvidenceRecord(
                text=text,
                evidence_class=EvidenceClass.VERIFIED_FACT,
                confidence=0.8,
                provenance_verified=True,
                source_reference=source_reference,
                accept_for_learning=True,
                rejection_reason=None,
                loss_target="L_KNOW",
            )

        return self._reject(text, EvidenceClass.QUERY, "NO_SIGNAL_DETECTED")

    def _matches_any(self, text: str, patterns: list) -> bool:
        return any(re.search(p, text) for p in patterns)

    def _reject(self, text: str, cls: EvidenceClass, reason: str) -> EvidenceRecord:
        return EvidenceRecord(
            text=text,
            evidence_class=cls,
            confidence=0.0,
            provenance_verified=False,
            source_reference=None,
            accept_for_learning=False,
            rejection_reason=reason,
            loss_target=None,
        )
