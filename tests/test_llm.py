import pytest
from langchain_core.exceptions import (
    ModelAPIError,
    ModelAuthenticationError,
    ModelInvalidRequestError,
    ModelRateLimitError,
)
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.retry import RunnableRetry
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import (
    GoogleAPIError,
    GoogleAuthenticationError,
    GoogleInvalidRequestError,
    GoogleModelNotFoundError,
    GooglePermissionDeniedError,
    GoogleRateLimitError,
)

from rag import RAGConfig
from rag.llm import TRANSIENT_ERRORS, build_llm, with_transient_retry


class FlakyLLM:
    """Raises the given error a fixed number of times, then answers.

    Wrapped in a RunnableLambda so it gets the real .with_retry() machinery,
    exactly as the Gemini client does.
    """

    def __init__(self, error: BaseException, failures: int):
        self.error = error
        self.failures = failures
        self.calls = 0

    def __call__(self, prompt):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        return f"answer to {prompt}"

    def as_runnable(self, max_attempts=3):
        # Zero backoff keeps the suite instant; the timing itself is tenacity's.
        return with_transient_retry(
            RunnableLambda(self), max_attempts=max_attempts, initial_wait=0, max_wait=0
        )


@pytest.mark.parametrize(
    "error",
    [ModelRateLimitError("429 RESOURCE_EXHAUSTED"), ModelAPIError("503 UNAVAILABLE")],
)
def test_transient_errors_are_retried_until_success(error):
    llm = FlakyLLM(error, failures=2)

    assert llm.as_runnable().invoke("q") == "answer to q"
    assert llm.calls == 3


def test_retries_stop_after_max_attempts():
    llm = FlakyLLM(ModelRateLimitError("429"), failures=10)

    with pytest.raises(ModelRateLimitError):
        llm.as_runnable(max_attempts=3).invoke("q")
    assert llm.calls == 3


@pytest.mark.parametrize(
    "error",
    [
        ModelAuthenticationError("401 bad key"),
        ModelInvalidRequestError("400 invalid argument"),
        ValueError("a bug in our own code"),
    ],
)
def test_non_transient_errors_are_not_retried(error):
    llm = FlakyLLM(error, failures=1)

    with pytest.raises(type(error)):
        llm.as_runnable().invoke("q")
    assert llm.calls == 1


@pytest.mark.parametrize("error_type", [GoogleRateLimitError, GoogleAPIError])
def test_gemini_429_and_5xx_errors_count_as_transient(error_type):
    assert issubclass(error_type, TRANSIENT_ERRORS)


@pytest.mark.parametrize(
    "error_type",
    [
        GoogleAuthenticationError,
        GooglePermissionDeniedError,
        GoogleInvalidRequestError,
        GoogleModelNotFoundError,
    ],
)
def test_gemini_client_errors_are_not_transient(error_type):
    assert not issubclass(error_type, TRANSIENT_ERRORS)


def test_build_llm_wraps_the_configured_model_in_a_bounded_retry(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-not-used")
    config = RAGConfig(llm_max_attempts=4, llm_retry_initial_wait=0.5, llm_retry_max_wait=5)

    llm = build_llm(config)

    assert isinstance(llm, RunnableRetry)
    assert isinstance(llm.bound, ChatGoogleGenerativeAI)
    assert llm.bound.model == config.llm_model
    # The client's own retries are off, so .with_retry() is the only layer.
    assert llm.bound.max_retries == 1
    assert llm.max_attempt_number == 4
    assert llm.retry_exception_types == TRANSIENT_ERRORS
    assert llm.exponential_jitter_params == {"initial": 0.5, "max": 5}


def test_build_llm_defaults_the_config(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-not-used")

    assert build_llm().bound.model == RAGConfig().llm_model
