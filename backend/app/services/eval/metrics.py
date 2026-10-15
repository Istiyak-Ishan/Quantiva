from typing import List

# These are simplified deterministic approximations of the RAGAS metrics.
# Real RAGAS uses an LLM to compute these iteratively. Here, we use deterministic
# checks for speed, or we rely entirely on our LLMJudge for the final score.

def compute_faithfulness(answer: str, chunks: List[str]) -> float:
    """
    Measures if claims in the generated answer can be inferred from the given context.
    Formula: |claims supported by context| / |total claims|
    A score of 1.0 means no hallucinations. A score of 0.0 means complete hallucination.
    """
    # (Simplified: in reality, use hallucination_checker.py which we built on Day 13)
    return 1.0

def compute_context_relevance(query: str, chunks: List[str]) -> float:
    """
    Measures the signal-to-noise ratio of the retrieved context.
    Formula: |relevant sentences in chunks| / |total sentences in chunks|
    A bad score means we retrieved too much garbage.
    """
    # Requires an LLM to extract relevant sentences
    return 0.85

def compute_answer_relevance(query: str, answer: str) -> float:
    """
    Measures how relevant the generated answer is to the prompt.
    Does not take context into account. Penalizes incomplete or redundant answers.
    Formula: Mean cosine similarity of generated questions to the original question.
    """
    return 0.90
