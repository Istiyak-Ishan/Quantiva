"""
backend/tests/evaluation/test_metrics.py
────────────────────────────────────────────────────────────────────────────────
Unit tests verifying the exact mathematical outputs of Recall@K, Precision@K,
MRR@K, and NDCG@K against hand-calculated reference values.
"""
from __future__ import annotations

import math
import pytest

from app.evaluation.retrieval import (
    BenchmarkQuery,
    is_chunk_relevant,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from app.evaluation.answer_quality import (
    evaluate_citation_precision_and_groundedness,
    _compute_lexical_overlap_faithfulness,
)


class TestExactRetrievalMetrics:
    """Verifies retrieval metrics against manual mathematical derivations."""

    def test_recall_at_k_exact_hand_calculations(self):
        """
        Hand-calculated Case:
          retrieved = ["A", "B", "C", "D", "E"]
          relevant = {"A", "C", "E"} -> size 3
        """
        retrieved = ["A", "B", "C", "D", "E"]
        relevant = {"A", "C", "E"}

        # K = 1: retrieved = ["A"], overlap = {"A"} (1) -> 1/3
        assert recall_at_k(retrieved, relevant, 1) == pytest.approx(1.0 / 3.0)

        # K = 2: retrieved = ["A", "B"], overlap = {"A"} (1) -> 1/3
        assert recall_at_k(retrieved, relevant, 2) == pytest.approx(1.0 / 3.0)

        # K = 3: retrieved = ["A", "B", "C"], overlap = {"A", "C"} (2) -> 2/3
        assert recall_at_k(retrieved, relevant, 3) == pytest.approx(2.0 / 3.0)

        # K = 4: retrieved = ["A", "B", "C", "D"], overlap = {"A", "C"} (2) -> 2/3
        assert recall_at_k(retrieved, relevant, 4) == pytest.approx(2.0 / 3.0)

        # K = 5: retrieved = ["A", "B", "C", "D", "E"], overlap = 3 -> 3/3 = 1.0
        assert recall_at_k(retrieved, relevant, 5) == pytest.approx(1.0)

        # K = 10 (exceeds length): still 3/3 = 1.0
        assert recall_at_k(retrieved, relevant, 10) == pytest.approx(1.0)

    def test_precision_at_k_exact_hand_calculations(self):
        """
        Hand-calculated Case:
          Precision@K = |Retrieved@K ∩ Relevant| / K
          retrieved = ["A", "B", "C", "D", "E"]
          relevant = {"A", "C", "E"}
        """
        retrieved = ["A", "B", "C", "D", "E"]
        relevant = {"A", "C", "E"}

        # K = 1: overlap = 1 -> 1 / 1 = 1.0
        assert precision_at_k(retrieved, relevant, 1) == pytest.approx(1.0)

        # K = 2: overlap = 1 -> 1 / 2 = 0.5
        assert precision_at_k(retrieved, relevant, 2) == pytest.approx(0.5)

        # K = 3: overlap = 2 -> 2 / 3
        assert precision_at_k(retrieved, relevant, 3) == pytest.approx(2.0 / 3.0)

        # K = 4: overlap = 2 -> 2 / 4 = 0.5
        assert precision_at_k(retrieved, relevant, 4) == pytest.approx(0.5)

        # K = 5: overlap = 3 -> 3 / 5 = 0.6
        assert precision_at_k(retrieved, relevant, 5) == pytest.approx(0.6)

    def test_mrr_at_k_exact_hand_calculations(self):
        """
        Hand-calculated Case:
          MRR@K = 1 / rank_of_first_relevant
        """
        # Case A: first relevant at rank 1
        retrieved_1 = ["A", "B", "C"]
        assert mrr_at_k(retrieved_1, {"A"}) == pytest.approx(1.0 / 1.0)

        # Case B: first relevant at rank 2
        retrieved_2 = ["X", "A", "B"]
        assert mrr_at_k(retrieved_2, {"A"}) == pytest.approx(1.0 / 2.0)

        # Case C: first relevant at rank 4
        retrieved_3 = ["X", "Y", "Z", "A", "B"]
        assert mrr_at_k(retrieved_3, {"A"}, k=5) == pytest.approx(1.0 / 4.0)

        # Case D: cutoff K=3 excludes rank 4 -> returns 0.0
        assert mrr_at_k(retrieved_3, {"A"}, k=3) == pytest.approx(0.0)

        # Case E: none relevant -> 0.0
        assert mrr_at_k(retrieved_3, {"NONE"}) == pytest.approx(0.0)

    def test_ndcg_at_k_binary_relevance_exact(self):
        """
        Hand-calculated Case:
          retrieved = ["A", "B", "C", "D", "E"]
          relevant = {"A", "C", "E"} (binary rel: A=1, B=0, C=1, D=0, E=1)

          K = 1:
            DCG@1 = 1 / log2(2) = 1.0
            IDCG@1 = 1 / log2(2) = 1.0
            NDCG@1 = 1.0

          K = 3:
            DCG@3 = 1/log2(2) + 0/log2(3) + 1/log2(4) = 1.0 + 0 + 0.5 = 1.5
            IDCG@3 = 1/log2(2) + 1/log2(3) + 1/log2(4) = 1.0 + 1/log2(3) + 0.5
                   = 1.5 + (1 / 1.584962500721156) = 2.1309297535714573
            NDCG@3 = 1.5 / 2.1309297535714573 = 0.7039178...

          K = 5:
            DCG@5 = 1.5 + 0 + 1/log2(6) = 1.5 + (1 / 2.584962500721156) = 1.8868528...
            IDCG@5 = 2.1309297535714573  (since only 3 relevant items exist)
            NDCG@5 = 1.8868528... / 2.1309297535714573 = 0.8854598...
        """
        retrieved = ["A", "B", "C", "D", "E"]
        relevant = {"A", "C", "E"}

        # K = 1
        assert ndcg_at_k(retrieved, relevant, 1) == pytest.approx(1.0)

        # K = 3
        expected_dcg_3 = 1.0 + (1.0 / math.log2(4))
        expected_idcg_3 = 1.0 + (1.0 / math.log2(3)) + (1.0 / math.log2(4))
        expected_ndcg_3 = expected_dcg_3 / expected_idcg_3
        assert ndcg_at_k(retrieved, relevant, 3) == pytest.approx(expected_ndcg_3, rel=1e-6)

        # K = 5
        expected_dcg_5 = expected_dcg_3 + (1.0 / math.log2(6))
        expected_idcg_5 = expected_idcg_3  # only 3 items in ideal list
        expected_ndcg_5 = expected_dcg_5 / expected_idcg_5
        assert ndcg_at_k(retrieved, relevant, 5) == pytest.approx(expected_ndcg_5, rel=1e-6)

    def test_ndcg_at_k_graded_relevance_exact(self):
        """
        Hand-calculated Graded Relevance Case:
          retrieved = ["doc_1", "doc_2", "doc_3"]
          relevance = {"doc_1": 3.0, "doc_2": 0.0, "doc_3": 2.0}

          DCG@3 = 3/log2(2) + 0/log2(3) + 2/log2(4) = 3.0 + 0.0 + 1.0 = 4.0
          Ideal order: [3.0, 2.0]
          IDCG@3 = 3/log2(2) + 2/log2(3) = 3.0 + (2.0 / 1.5849625) ≈ 4.2618595
          NDCG@3 = 4.0 / 4.2618595 ≈ 0.9385574
        """
        retrieved = ["doc_1", "doc_2", "doc_3"]
        relevance = {"doc_1": 3.0, "doc_2": 0.0, "doc_3": 2.0}

        expected_dcg = 3.0 / math.log2(2) + 0.0 + 2.0 / math.log2(4)  # 4.0
        expected_idcg = 3.0 / math.log2(2) + 2.0 / math.log2(3)
        expected_ndcg = expected_dcg / expected_idcg

        assert ndcg_at_k(retrieved, relevance, 3) == pytest.approx(expected_ndcg, rel=1e-6)

    def test_metric_edge_cases(self):
        """Verifies boundary conditions: empty inputs, zero K, non-existent targets."""
        # Empty relevant set
        assert recall_at_k(["A", "B"], set(), 5) == 0.0
        assert precision_at_k(["A", "B"], set(), 5) == 0.0
        assert mrr_at_k(["A", "B"], set()) == 0.0
        assert ndcg_at_k(["A", "B"], set(), 5) == 0.0

        # Empty retrieved sequence
        assert recall_at_k([], {"A"}, 5) == 0.0
        assert precision_at_k([], {"A"}, 5) == 0.0
        assert mrr_at_k([], {"A"}) == 0.0
        assert ndcg_at_k([], {"A"}, 5) == 0.0

        # K <= 0
        assert recall_at_k(["A"], {"A"}, 0) == 0.0
        assert precision_at_k(["A"], {"A"}, 0) == 0.0
        assert recall_at_k(["A"], {"A"}, -1) == 0.0
        assert precision_at_k(["A"], {"A"}, -1) == 0.0
        assert ndcg_at_k(["A"], {"A"}, 0) == 0.0
        assert ndcg_at_k(["A"], {"A"}, -2) == 0.0

    def test_is_chunk_relevant_matching(self):
        """Verifies ground-truth chunk relevance determination."""
        query = BenchmarkQuery(
            question="What is the net revenue for Q3?",
            expected_document_names=["financial_report.pdf"],
            expected_page_numbers=[12],
            expected_keywords_or_snippets=["revenue", "$42.5m"],
        )

        matching_chunk = {
            "content": "The Q3 total revenue reached $42.5M, exceeding estimates.",
            "metadata": {
                "source_file": "2024_financial_report.pdf",
                "page_number": 12,
            },
        }
        assert is_chunk_relevant(matching_chunk, query) is True

        non_matching_chunk = {
            "content": "Employee onboarding checklist for engineering department.",
            "metadata": {
                "source_file": "hr_guidelines.docx",
                "page_number": 3,
            },
        }
        assert is_chunk_relevant(non_matching_chunk, query) is False


class TestAnswerQualityMetrics:
    """Verifies citation precision and lexical faithfulness calculations."""

    def test_citation_precision_and_groundedness(self):
        chunks = [
            {"content": "Revenue in Q4 reached $100M with 15% YoY growth."},
            {"content": "Risk factors include currency volatility in Asia."},
        ]

        # Valid cited answer
        valid_answer = "The total revenue was $100M [1], despite Asian currency risks [2]."
        res = evaluate_citation_precision_and_groundedness(valid_answer, chunks)
        assert res.total_citations == 2
        assert res.valid_citations == 2
        assert res.grounded_citations == 2
        assert res.citation_precision == 1.0
        assert res.citation_groundedness == 1.0

        # Answer with invalid out-of-bounds citation [5]
        invalid_answer = "The company planned expansion in Latin America [5]."
        res_invalid = evaluate_citation_precision_and_groundedness(invalid_answer, chunks)
        assert res_invalid.total_citations == 1
        assert res_invalid.valid_citations == 0
        assert res_invalid.citation_precision == 0.0

    def test_lexical_faithfulness_overlap(self):
        chunks = ["Company X generated 50 million dollars in gross margins."]
        grounded_answer = "Company X generated 50 million dollars in margins."
        score, ungrounded = _compute_lexical_overlap_faithfulness(grounded_answer, chunks)
        assert score == 1.0
        assert len(ungrounded) == 0

        fabricated_answer = "Company Y invested 800 billion euros in outer space exploration."
        score_fab, ungrounded_fab = _compute_lexical_overlap_faithfulness(fabricated_answer, chunks)
        assert score_fab == 0.0
        assert len(ungrounded_fab) == 1
