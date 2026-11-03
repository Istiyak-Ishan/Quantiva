"""
backend/app/core/metrics.py
────────────────────────────────────────────────────────────────────────────────
Prometheus metrics registry for Quantiva.

Exposed metrics:
  • http_requests_total           Counter   – request count by endpoint + status
  • http_request_duration_seconds Histogram – latency distribution by endpoint
  • llm_tokens_total              Counter   – tokens consumed by model + direction
  • llm_request_duration_seconds  Histogram – end-to-end LLM call latency
  • llm_cost_usd_total            Counter   – estimated LLM API cost in USD
  • cache_hits_total              Counter   – semantic cache hits
  • cache_misses_total            Counter   – semantic cache misses
  • active_streaming_sessions     Gauge     – concurrent SSE streams open
  • documents_ingested_total      Counter   – documents fully indexed
  • retrieval_duration_seconds    Histogram – hybrid retrieval latency

Usage:
  from app.core.metrics import (
      HTTP_REQUESTS, HTTP_LATENCY,
      LLM_TOKENS, LLM_LATENCY, LLM_COST,
      CACHE_HITS, CACHE_MISSES,
      ACTIVE_STREAMS, DOCS_INGESTED,
      RETRIEVAL_LATENCY,
  )
"""

from prometheus_client import Counter, Histogram, Gauge, REGISTRY  # noqa: F401
import prometheus_client  # noqa: F401

# ── HTTP layer ────────────────────────────────────────────────────────────────

HTTP_REQUESTS = Counter(
    "http_requests_total",
    "Total HTTP requests received",
    ["method", "endpoint", "status_code"],
)

HTTP_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

# ── LLM layer ─────────────────────────────────────────────────────────────────

LLM_TOKENS = Counter(
    "llm_tokens_total",
    "Total tokens sent/received from the LLM",
    ["model", "direction"],   # direction: input | output
)

LLM_LATENCY = Histogram(
    "llm_request_duration_seconds",
    "End-to-end LLM generation latency in seconds",
    ["model"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

LLM_COST = Counter(
    "llm_cost_usd_total",
    "Estimated LLM API cost in USD",
    ["model"],
)

# ── Cache layer ───────────────────────────────────────────────────────────────

CACHE_HITS = Counter(
    "cache_hits_total",
    "Semantic cache hits (query answered without LLM call)",
    ["cache_type"],           # cache_type: semantic | exact
)

CACHE_MISSES = Counter(
    "cache_misses_total",
    "Semantic cache misses (LLM call required)",
    ["cache_type"],
)

# ── Application layer ──────────────────────────────────────────────────────────

ACTIVE_STREAMS = Gauge(
    "active_streaming_sessions",
    "Number of currently open SSE streaming connections",
)

DOCS_INGESTED = Counter(
    "documents_ingested_total",
    "Total documents fully indexed",
    ["file_type"],            # file_type: pdf | xlsx | csv | docx
)

RETRIEVAL_LATENCY = Histogram(
    "retrieval_duration_seconds",
    "Hybrid retrieval (vector + BM25 + rerank) latency",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
)
