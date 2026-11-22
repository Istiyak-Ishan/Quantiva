"""
backend/app/evaluation/answer_quality.py
────────────────────────────────────────────────────────────────────────────────
RAG Answer Quality Evaluation & Performance Telemetry (Prompt P8.1).

Measures:
  • Citation Precision & Groundedness:
      Verifies that each citation [n] resolves to a valid retrieved chunk
      and that the chunk supports the cited facts/numbers.
  • Answer Faithfulness:
      Deterministic lexical/claim overlap + optional LLM-as-a-Judge scoring.
  • Performance Telemetry:
      Retrieval latency, reranking latency, TTFT, total generation latency,
      ingestion throughput (pages/sec, chunks/sec), and peak RSS memory delta.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
import json
import logging
import os
import re
import time
from typing import Any

import psutil

from app.providers.interfaces.llm import LLMProvider
from app.providers.interfaces.types import ChatMessage, ChatRole, GenerationConfig
from app.retrieval.types import RetrievedChunk

logger = logging.getLogger(__name__)


# ─── TELEMETRY & MEMORY MEASUREMENT ──────────────────────────────────────────

def get_current_rss_mb() -> float:
    """Returns current process Resident Set Size (RSS) in Megabytes."""
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        return 0.0


@dataclass
class PerformanceTelemetry:
    retrieval_latency_ms: float = 0.0
    reranking_latency_ms: float = 0.0
    ttft_ms: float | None = None
    total_generation_latency_ms: float = 0.0
    total_pipeline_latency_ms: float = 0.0
    ingestion_throughput_pages_per_sec: float | None = None
    ingestion_throughput_chunks_per_sec: float | None = None
    peak_rss_mb_delta: float = 0.0
    start_rss_mb: float = 0.0
    end_rss_mb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieval_latency_ms": round(self.retrieval_latency_ms, 2),
            "reranking_latency_ms": round(self.reranking_latency_ms, 2),
            "ttft_ms": round(self.ttft_ms, 2) if self.ttft_ms is not None else None,
            "total_generation_latency_ms": round(self.total_generation_latency_ms, 2),
            "total_pipeline_latency_ms": round(self.total_pipeline_latency_ms, 2),
            "ingestion_throughput_pages_per_sec": (
                round(self.ingestion_throughput_pages_per_sec, 2)
                if self.ingestion_throughput_pages_per_sec is not None
                else None
            ),
            "ingestion_throughput_chunks_per_sec": (
                round(self.ingestion_throughput_chunks_per_sec, 2)
                if self.ingestion_throughput_chunks_per_sec is not None
                else None
            ),
            "peak_rss_mb_delta": round(self.peak_rss_mb_delta, 2),
            "start_rss_mb": round(self.start_rss_mb, 2),
            "end_rss_mb": round(self.end_rss_mb, 2),
        }


class TelemetryTracker:
    """Context tracker for measuring execution time and RSS memory usage."""

    def __init__(self) -> None:
        self.telemetry = PerformanceTelemetry()
        self._start_time: float = 0.0

    def __enter__(self) -> TelemetryTracker:
        self.telemetry.start_rss_mb = get_current_rss_mb()
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        total_sec = time.perf_counter() - self._start_time
        self.telemetry.total_pipeline_latency_ms = total_sec * 1000.0
        self.telemetry.end_rss_mb = get_current_rss_mb()
        self.telemetry.peak_rss_mb_delta = max(
            0.0, self.telemetry.end_rss_mb - self.telemetry.start_rss_mb
        )


# ─── CITATION PRECISION & GROUNDEDNESS EVALUATION ────────────────────────────

@dataclass
class CitationEvaluationDetail:
    marker: str
    chunk_index: int
    is_valid_index: bool
    is_grounded: bool
    reason: str
    cited_sentence: str


@dataclass
class CitationEvaluationResult:
    total_citations: int
    valid_citations: int
    grounded_citations: int
    citation_precision: float
    citation_groundedness: float
    details: list[CitationEvaluationDetail] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_citations": self.total_citations,
            "valid_citations": self.valid_citations,
            "grounded_citations": self.grounded_citations,
            "citation_precision": round(self.citation_precision, 4),
            "citation_groundedness": round(self.citation_groundedness, 4),
            "details": [asdict(d) for d in self.details],
        }


def _extract_chunk_text(chunk: RetrievedChunk | dict[str, Any] | str) -> str:
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, RetrievedChunk):
        return chunk.content
    return str(chunk.get("content", ""))


def evaluate_citation_precision_and_groundedness(
    answer: str,
    retrieved_chunks: Sequence[RetrievedChunk | dict[str, Any] | str],
) -> CitationEvaluationResult:
    """
    Evaluates every citation marker [n] in the answer:
      - Valid index: 1 <= n <= len(retrieved_chunks)
      - Groundedness: Checks if chunk[n-1] contains the numbers or key entities
        present in the sentence citing [n].
    """
    if not answer.strip():
        return CitationEvaluationResult(
            total_citations=0,
            valid_citations=0,
            grounded_citations=0,
            citation_precision=1.0,
            citation_groundedness=1.0,
            details=[],
        )

    # Split text into sentences for context
    sentence_delimiters = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_delimiters.split(answer)

    citation_pattern = re.compile(r'\[(\d+)\]')
    details: list[CitationEvaluationDetail] = []
    num_chunks = len(retrieved_chunks)

    for sentence in sentences:
        citation_matches = list(citation_pattern.finditer(sentence))
        if not citation_matches:
            continue

        prev_end = 0
        for m in citation_matches:
            idx = int(m.group(1))
            marker = f"[{idx}]"
            is_valid = 1 <= idx <= num_chunks

            if not is_valid:
                details.append(
                    CitationEvaluationDetail(
                        marker=marker,
                        chunk_index=idx,
                        is_valid_index=False,
                        is_grounded=False,
                        reason=f"Citation index {idx} exceeds retrieved chunks count ({num_chunks})",
                        cited_sentence=sentence.strip(),
                    )
                )
                prev_end = m.end()
                continue

            # Extract local clause preceding this citation marker
            local_clause = sentence[prev_end:m.start()].strip()
            prev_end = m.end()

            # If local clause is too short (e.g. stacked citations [1][2]), fallback to full sentence
            clause_to_evaluate = local_clause if len(local_clause) > 10 else sentence

            words = set(re.findall(r'\b[a-zA-Z0-9_]{3,}\b', clause_to_evaluate.lower()))
            numbers = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', clause_to_evaluate))

            # Target chunk (0-indexed)
            target_chunk = retrieved_chunks[idx - 1]
            chunk_text = _extract_chunk_text(target_chunk).lower()

            # Check number grounding (numbers in clause should ideally appear in chunk)
            nums_present = [num for num in numbers if num in chunk_text]
            number_match = True if not numbers else len(nums_present) > 0

            # Check lexical keyword overlap
            words_in_chunk = [w for w in words if w in chunk_text]
            word_overlap = len(words_in_chunk) / max(1, len(words))

            is_grounded = (number_match and word_overlap >= 0.2) or len(words_in_chunk) >= 2
            reason = (
                "Verified"
                if is_grounded
                else f"Low grounding overlap: {len(words_in_chunk)}/{len(words)} keywords found"
            )

            details.append(
                CitationEvaluationDetail(
                    marker=marker,
                    chunk_index=idx,
                    is_valid_index=True,
                    is_grounded=is_grounded,
                    reason=reason,
                    cited_sentence=sentence.strip(),
                )
            )

    total_citations = len(details)
    if total_citations == 0:
        # If no citations were present in answer
        return CitationEvaluationResult(
            total_citations=0,
            valid_citations=0,
            grounded_citations=0,
            citation_precision=1.0,
            citation_groundedness=1.0,
            details=[],
        )

    valid_count = sum(1 for d in details if d.is_valid_index)
    grounded_count = sum(1 for d in details if d.is_grounded)

    precision = valid_count / float(total_citations)
    groundedness = grounded_count / float(total_citations)

    return CitationEvaluationResult(
        total_citations=total_citations,
        valid_citations=valid_count,
        grounded_citations=grounded_count,
        citation_precision=precision,
        citation_groundedness=groundedness,
        details=details,
    )


# ─── ANSWER FAITHFULNESS EVALUATION ──────────────────────────────────────────

@dataclass
class FaithfulnessResult:
    faithfulness_score: float
    lexical_overlap_score: float
    llm_judge_score: float | None
    judge_reasoning: str | None
    ungrounded_claims: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "faithfulness_score": round(self.faithfulness_score, 4),
            "lexical_overlap_score": round(self.lexical_overlap_score, 4),
            "llm_judge_score": (
                round(self.llm_judge_score, 4) if self.llm_judge_score is not None else None
            ),
            "judge_reasoning": self.judge_reasoning,
            "ungrounded_claims": self.ungrounded_claims,
        }


def _compute_lexical_overlap_faithfulness(
    answer: str,
    context_chunks: Sequence[RetrievedChunk | dict[str, Any] | str],
) -> tuple[float, list[str]]:
    """
    Computes sentence-by-sentence lexical and factual entity overlap against context.
    Returns (overlap_score_in_0_to_1, ungrounded_sentences).
    """
    if not answer.strip():
        return 1.0, []

    full_context_text = " ".join(
        _extract_chunk_text(ch) for ch in context_chunks
    ).lower()

    if not full_context_text.strip():
        return 0.0, [answer]

    sentence_delimiters = re.compile(r'(?<=[.!?])\s+')
    sentences = [
        s.strip() for s in sentence_delimiters.split(answer) if len(s.strip()) > 5
    ]
    if not sentences:
        return 1.0, []

    ungrounded: list[str] = []
    grounded_count = 0

    for s in sentences:
        # Extract meaningful terms (length >= 3)
        terms = set(re.findall(r'\b[a-zA-Z0-9_]{3,}\b', s.lower()))
        numbers = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', s))

        # Check if numbers are present in context
        nums_present = [n for n in numbers if n in full_context_text]
        nums_ok = True if not numbers else (len(nums_present) / len(numbers)) >= 0.5

        # Check keyword overlap
        terms_present = [t for t in terms if t in full_context_text]
        terms_ratio = (len(terms_present) / max(1, len(terms)))

        if nums_ok and terms_ratio >= 0.4:
            grounded_count += 1
        else:
            ungrounded.append(s)

    overlap_score = grounded_count / float(len(sentences))
    return overlap_score, ungrounded


async def evaluate_answer_faithfulness(
    answer: str,
    context_chunks: Sequence[RetrievedChunk | dict[str, Any] | str],
    llm_provider: LLMProvider | None = None,
) -> FaithfulnessResult:
    """
    Measures faithfulness using:
      1. Deterministic lexical/numerical claim overlap.
      2. Optional LLM-as-a-Judge scoring (0.0 to 1.0) using the active LLM.
    """
    lexical_score, ungrounded_claims = _compute_lexical_overlap_faithfulness(
        answer, context_chunks
    )

    llm_judge_score: float | None = None
    judge_reasoning: str | None = None

    if llm_provider is not None and answer.strip():
        context_excerpt = "\n\n".join(
            f"[Chunk {i+1}]: {_extract_chunk_text(c)}"
            for i, c in enumerate(context_chunks[:5])
        )

        judge_prompt = (
            "You are an impartial RAG evaluation judge. Evaluate whether the provided Answer is "
            "completely faithful to and supported by the Context.\n\n"
            f"--- CONTEXT ---\n{context_excerpt}\n\n"
            f"--- ANSWER ---\n{answer}\n\n"
            "Evaluate:\n"
            "1. Does the answer contain any hallucinations or ungrounded claims not supported by the context?\n"
            "2. Score faithfulness strictly from 0.0 (completely fabricated) to 1.0 (completely grounded).\n\n"
            "Respond ONLY with a JSON object in this exact schema:\n"
            '{"score": <float between 0.0 and 1.0>, "reasoning": "<concise explanation>", "ungrounded_claims": ["claim1", ...]}'
        )

        try:
            response = await llm_provider.generate_content(
                messages=[ChatMessage(role=ChatRole.USER, content=judge_prompt)],
                config=GenerationConfig(temperature=0.0, max_tokens=512),
            )
            raw_text = response.text.strip()
            # Clean json fences if present
            cleaned = re.sub(r'^```(?:json)?\s*', '', raw_text, flags=re.MULTILINE)
            cleaned = re.sub(r'```$', '', cleaned, flags=re.MULTILINE).strip()

            parsed = json.loads(cleaned)
            score_val = float(parsed.get("score", 0.0))
            llm_judge_score = max(0.0, min(1.0, score_val))
            judge_reasoning = str(parsed.get("reasoning", ""))
            judge_ungrounded = parsed.get("ungrounded_claims", [])
            if isinstance(judge_ungrounded, list) and judge_ungrounded:
                ungrounded_claims = [str(c) for c in judge_ungrounded]
        except Exception as exc:
            logger.warning("LLM-as-a-judge evaluation failed, using lexical overlap: %s", exc)

    # Composite faithfulness score
    if llm_judge_score is not None:
        final_faithfulness = (0.7 * llm_judge_score) + (0.3 * lexical_score)
    else:
        final_faithfulness = lexical_score

    return FaithfulnessResult(
        faithfulness_score=final_faithfulness,
        lexical_overlap_score=lexical_score,
        llm_judge_score=llm_judge_score,
        judge_reasoning=judge_reasoning,
        ungrounded_claims=ungrounded_claims,
    )
