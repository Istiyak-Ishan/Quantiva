"""Structured citation attribution and lexical evidence verification."""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.retrieval.types import RetrievedChunk, chunk_from_hit

_CITATION_RE = re.compile(r"\[(\d+)\]")
_QUOTE_RE = re.compile(r'["“](.+?)["”]')
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_STOP_WORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "have", "in", "is", "it", "of", "on", "or", "that", "the",
        "this", "to", "was", "were", "will", "with",
    }
)


@dataclass(frozen=True, slots=True)
class StructuredCitation:
    marker: str
    index: int
    chunk_id: str
    document_id: str
    source_file: str
    page_number: int | str | None
    quote_snippet: str
    lexical_overlap: float
    quote_verified: bool
    valid: bool


@dataclass(slots=True)
class CitationValidationResult:
    citations: list[StructuredCitation] = field(default_factory=list)
    phantom_citations: list[int] = field(default_factory=list)
    unsupported_citations: list[int] = field(default_factory=list)
    sanitized_answer: str = ""

    @property
    def valid(self) -> bool:
        return not self.phantom_citations and not self.unsupported_citations

    @property
    def is_valid(self) -> bool:
        return self.valid

    @property
    def hallucinated_citations(self) -> list[int]:
        """Alias covering phantom and unsupported citation indices."""
        return list(dict.fromkeys([*self.phantom_citations, *self.unsupported_citations]))

    @property
    def invalid_citations(self) -> list[int]:
        return self.hallucinated_citations


def _tokens(text: str) -> set[str]:
    return {
        token.casefold()
        for token in _TOKEN_RE.findall(text or "")
        if len(token) > 1 and token.casefold() not in _STOP_WORDS
    }


def _claim_before_marker(answer: str, marker_start: int) -> str:
    prefix = answer[:marker_start]
    boundaries = [prefix.rfind(char) for char in ".!?\n"]
    return prefix[max(boundaries) + 1 :].strip()


def _best_verbatim_snippet(content: str, claim_tokens: set[str], limit: int) -> str:
    candidates = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", content)]
    candidates = [part for part in candidates if part]
    if not candidates:
        return content[:limit]
    best = max(
        candidates,
        key=lambda part: len(_tokens(part) & claim_tokens),
    )
    return best[:limit]


def _normalise_chunk(value: RetrievedChunk | Mapping[str, Any]) -> RetrievedChunk:
    if isinstance(value, RetrievedChunk):
        return value
    # Canonical chunk mappings carry metadata fields at their top level rather
    # than under ``payload``.  Preserve those fields for attribution.
    if "metadata" not in value and "payload" not in value:
        metadata = dict(value)
        wrapped = {
            "chunk_id": value.get("chunk_id") or value.get("id"),
            "content": value.get("content") or value.get("text") or "",
            "score": value.get("score", 0.0),
            "metadata": metadata,
        }
        return chunk_from_hit(wrapped)
    return chunk_from_hit(value)


class CitationValidator:
    """Validate numeric answer markers against the retrieved context window."""

    def __init__(self, *, minimum_overlap: float = 0.1, snippet_length: int = 300) -> None:
        if not 0 <= minimum_overlap <= 1:
            raise ValueError("minimum_overlap must be between 0 and 1")
        self.minimum_overlap = float(minimum_overlap)
        self.snippet_length = int(snippet_length)

    def validate_citations(
        self,
        answer_text: str,
        retrieved_chunks: Sequence[RetrievedChunk | Mapping[str, Any]],
        *,
        strip_invalid: bool = False,
    ) -> CitationValidationResult:
        chunks = [_normalise_chunk(value) for value in retrieved_chunks]
        citations: list[StructuredCitation] = []
        phantom: list[int] = []
        unsupported: list[int] = []
        invalid_markers: set[str] = set()
        seen: set[int] = set()

        for match in _CITATION_RE.finditer(answer_text or ""):
            index = int(match.group(1))
            if index in seen:
                continue
            seen.add(index)
            marker = match.group(0)
            if index <= 0 or index > len(chunks):
                phantom.append(index)
                invalid_markers.add(marker)
                continue

            chunk = chunks[index - 1]
            claim = _claim_before_marker(answer_text, match.start())
            claim_tokens = _tokens(claim)
            content_tokens = _tokens(chunk.content)
            overlap = (
                len(claim_tokens & content_tokens) / len(claim_tokens)
                if claim_tokens
                else 0.0
            )
            quoted_phrases = _QUOTE_RE.findall(claim)
            quote_verified = any(
                phrase.strip().casefold() in chunk.content.casefold()
                for phrase in quoted_phrases
                if phrase.strip()
            )
            valid = quote_verified or overlap >= self.minimum_overlap
            if not valid:
                unsupported.append(index)
                invalid_markers.add(marker)

            metadata = chunk.metadata
            document_id = str(
                metadata.get("document_id")
                or metadata.get("doc_id")
                or ""
            )
            source_file = str(
                metadata.get("source_file")
                or metadata.get("file_name")
                or metadata.get("filename")
                or "unknown"
            )
            page_number = metadata.get("page_number", metadata.get("page"))
            quote_snippet = _best_verbatim_snippet(
                chunk.content,
                claim_tokens,
                self.snippet_length,
            )
            citations.append(
                StructuredCitation(
                    marker=marker,
                    index=index,
                    chunk_id=chunk.chunk_id,
                    document_id=document_id,
                    source_file=source_file,
                    page_number=page_number,
                    quote_snippet=quote_snippet,
                    lexical_overlap=round(overlap, 6),
                    quote_verified=quote_verified,
                    valid=valid,
                )
            )

        sanitized = answer_text
        if strip_invalid:
            for marker in invalid_markers:
                sanitized = sanitized.replace(marker, "")
            sanitized = re.sub(r"[ \t]{2,}", " ", sanitized)

        return CitationValidationResult(
            citations=citations,
            phantom_citations=phantom,
            unsupported_citations=unsupported,
            sanitized_answer=sanitized,
        )


__all__ = [
    "CitationValidationResult",
    "CitationValidator",
    "StructuredCitation",
]
