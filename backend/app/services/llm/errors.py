"""Safe errors exposed to API callers; provider exception details stay private."""


class ProviderError(RuntimeError):
    code = "provider_unavailable"
    status_code = 503


class LLMUnavailableError(ProviderError):
    def __init__(self):
        super().__init__("The language model is unavailable. Please retry later.")


class EmbeddingError(ProviderError):
    code = "embedding_unavailable"

    def __init__(self):
        super().__init__("Embedding generation failed. No substitute vectors were created.")


class ModelResponseError(ProviderError):
    code = "invalid_model_response"
    status_code = 502

    def __init__(self):
        super().__init__("The model returned an invalid response. Please retry.")


class AgentStepLimitError(ProviderError):
    """The agent used every tool step without producing a final answer."""

    code = "agent_step_limit"
    status_code = 502

    def __init__(self):
        super().__init__(
            "The assistant could not finish within the allowed number of steps. "
            "Try a more specific question."
        )
