"""Construction of the chat model, with retries for transient API failures.

The Gemini API sheds load with 429 (rate limited) and 5xx (overloaded)
responses that usually succeed on a second try. The model is wrapped in
LangChain's .with_retry() so those are retried with exponential backoff and
jitter, while real errors (bad key, invalid request, unknown model) fail
fast on the first attempt.
"""

from langchain_core.exceptions import ModelAPIError, ModelRateLimitError
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI

from .config import RAGConfig

# Provider-agnostic LangChain error types. langchain-google-genai raises 429 as
# GoogleRateLimitError (a ModelRateLimitError) and every 5xx, including 503,
# as GoogleAPIError (a ModelAPIError). Auth, permission, 400 and 404 errors
# are separate ModelError subclasses, so they are never retried.
TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (ModelRateLimitError, ModelAPIError)


def with_transient_retry(
    llm: Runnable,
    max_attempts: int = 3,
    initial_wait: float = 1.0,
    max_wait: float = 10.0,
    retry_on: tuple[type[BaseException], ...] = TRANSIENT_ERRORS,
) -> Runnable:
    """Wrap any runnable so transient failures are retried with backoff.

    max_attempts counts the first call, so 3 means up to two retries.
    """
    return llm.with_retry(
        retry_if_exception_type=retry_on,
        wait_exponential_jitter=True,
        exponential_jitter_params={"initial": initial_wait, "max": max_wait},
        stop_after_attempt=max_attempts,
    )


def with_config_retry(runnable: Runnable, config: RAGConfig | None = None) -> Runnable:
    """with_transient_retry, with attempts and waits taken from RAGConfig."""
    config = config or RAGConfig()
    return with_transient_retry(
        runnable,
        max_attempts=config.llm_max_attempts,
        initial_wait=config.llm_retry_initial_wait,
        max_wait=config.llm_retry_max_wait,
    )


def build_chat_model(config: RAGConfig | None = None) -> ChatGoogleGenerativeAI:
    """The bare configured Gemini chat model, without the retry wrapper.

    For callers that need the chat-model interface itself, e.g. bind_tools()
    for the agent, which a retry-wrapped runnable no longer exposes. Such
    callers apply with_config_retry() to whatever they build on top of it.
    """
    config = config or RAGConfig()
    # The client retries internally by default (max_retries=6), which would
    # multiply with .with_retry() into up to 18 HTTP calls. One attempt here
    # leaves .with_retry() as the single, bounded retry layer.
    return ChatGoogleGenerativeAI(model=config.llm_model, max_retries=1)


def build_llm(config: RAGConfig | None = None) -> Runnable:
    """The configured Gemini chat model, retried on transient failures."""
    return with_config_retry(build_chat_model(config), config)
