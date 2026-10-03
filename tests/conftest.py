"""Shared fixtures and offline test doubles.

Nothing here touches the network or needs an API key: the embedding model
and the LLM are both replaced with deterministic fakes, which is only
possible because KnowledgeBase and RAGPipeline take them as arguments.
"""

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from rag import RAGConfig, RAGPipeline


class HashEmbeddings(Embeddings):
    """Deterministic offline stand-in for the Hugging Face Inference API.

    Real embeddings are not reproducible across runs and need a token;
    this hashes characters into a fixed-width vector so similar strings
    land near each other, which is all the retrieval tests require.
    """

    dimensions = 16

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for index, char in enumerate(text):
            vector[index % self.dimensions] += ord(char) % 7
        return vector

    def embed_documents(self, texts):
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)


class FakeRetriever:
    """Returns a fixed set of documents and records what it was asked."""

    def __init__(self, documents):
        self._documents = documents
        self.last_question = None

    def invoke(self, question):
        self.last_question = question
        return self._documents


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    """Records the prompt it was handed so tests can assert on the context."""

    def __init__(self, reply="stub answer"):
        self.reply = reply
        self.last_prompt = None
        self.call_count = 0

    def invoke(self, prompt):
        self.last_prompt = prompt
        self.call_count += 1
        return FakeMessage(self.reply)

    @property
    def rendered_prompt(self) -> str:
        return self.last_prompt.to_string()


class ScriptedChatModel(BaseChatModel):
    """A tool-calling chat model that replays a fixed script of replies.

    Each call pops the next AIMessage, so a test can script "call tool X
    with these args" followed by a final answer, and drive a real
    AgentExecutor end to end without a network. Every prompt it receives is
    recorded so tests can assert that tool results were fed back to it.
    """

    script: list[AIMessage]
    prompts: list = Field(default_factory=list)
    bound_tools: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = [tool.name for tool in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.prompts.append(messages)
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])


def tool_call(name: str, call_id: str = "call_1", **args) -> AIMessage:
    """An assistant turn that asks the agent to run one tool."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


class RaisingRedis:
    """A Redis stand-in whose every command raises, to drive the cache's
    graceful-degradation branches without a real (mis)configured server."""

    def get(self, *args, **kwargs):
        raise ConnectionError("redis is down")

    def set(self, *args, **kwargs):
        raise ConnectionError("redis is down")


@pytest.fixture
def fake_redis():
    """A fresh in-memory Redis for each test, so the suite stays offline."""
    import fakeredis

    return fakeredis.FakeStrictRedis()


@pytest.fixture
def embeddings():
    return HashEmbeddings()


@pytest.fixture
def sample_documents():
    return [
        Document(page_content="Gradient descent optimises the loss.", metadata={"page": 1}),
        Document(page_content="Backpropagation computes gradients.", metadata={"page": 2}),
        Document(page_content="Transformers use self attention.", metadata={"page": 3}),
    ]


@pytest.fixture
def config(tmp_path):
    """A config isolated to this test's temporary directory."""
    return RAGConfig(persist_directory=str(tmp_path / "store"))


@pytest.fixture
def make_pipeline():
    """Builds a RAGPipeline over fakes, returning it alongside the fake LLM."""

    def _make(documents, reply="stub answer", **kwargs):
        llm = FakeLLM(reply)
        retriever = FakeRetriever(documents)
        pipeline = RAGPipeline(retriever=retriever, llm=llm, **kwargs)
        return pipeline, llm, retriever

    return _make


@pytest.fixture
def tiny_pdf(tmp_path):
    """A real single-page PDF, so the PyPDFLoader path is genuinely exercised."""
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas

    path = tmp_path / "tiny.pdf"
    pdf = canvas.Canvas(str(path), pagesize=LETTER)
    pdf.drawString(72, 720, "Deep learning uses gradient descent.")
    pdf.showPage()
    pdf.drawString(72, 720, "Attention is computed over token pairs.")
    pdf.showPage()
    pdf.save()
    return path
