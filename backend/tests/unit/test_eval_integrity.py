from unittest.mock import patch

import pytest

from app.services.eval.llm_judge import LLMJudge
from app.services.llm.errors import LLMUnavailableError, ModelResponseError


def test_judge_failure_never_returns_random_score():
    judge = LLMJudge()
    with patch.object(judge.client, "complete", side_effect=LLMUnavailableError()):
        with pytest.raises(LLMUnavailableError):
            judge.score("question", "context", "answer")


def test_judge_malformed_json_is_failure():
    judge = LLMJudge()
    with patch.object(judge.client, "complete", return_value="not JSON"):
        with pytest.raises(ModelResponseError):
            judge.score("question", "context", "answer")
