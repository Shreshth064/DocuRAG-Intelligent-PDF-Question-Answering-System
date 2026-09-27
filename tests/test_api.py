import io

import pytest

from rag import KnowledgeBase, RAGConfig
from rag.api import create_app
from rag.pipeline import NOT_FOUND_MESSAGE


@pytest.fixture
def app(config, embeddings):
    """A fully offline app: real KnowledgeBase over fake embeddings, fake LLM."""
    from conftest import FakeLLM

    knowledge_base = KnowledgeBase(config, embeddings=embeddings)
    return create_app(config=config, knowledge_base=knowledge_base, llm=FakeLLM("stub answer"))


@pytest.fixture
def client(app):
    return app.test_client()


def _pdf_bytes(text="Deep learning uses gradient descent. " * 40) -> bytes:
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=LETTER)
    pdf.drawString(72, 720, text[:120])
    pdf.showPage()
    pdf.save()
    return buffer.getvalue()


# --- /health ---

def test_health_is_ok_with_no_dependencies(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


# --- /upload ---

def test_upload_missing_file_is_400(client):
    response = client.post("/upload", data={}, content_type="multipart/form-data")

    assert response.status_code == 400
    assert "error" in response.get_json()


def test_upload_non_pdf_is_415(client):
    data = {"file": (io.BytesIO(b"this is plain text, not a pdf"), "notes.txt")}
    response = client.post("/upload", data=data, content_type="multipart/form-data")

    assert response.status_code == 415
    assert response.get_json()["error"] == "not a PDF"


def test_upload_disguised_non_pdf_is_415(client):
    # A .pdf extension is not enough; the bytes must start with %PDF-.
    data = {"file": (io.BytesIO(b"nope"), "fake.pdf")}
    response = client.post("/upload", data=data, content_type="multipart/form-data")

    assert response.status_code == 415


def test_upload_valid_pdf_is_201_with_chunk_count(client):
    data = {"file": (io.BytesIO(_pdf_bytes()), "book.pdf")}
    response = client.post("/upload", data=data, content_type="multipart/form-data")

    assert response.status_code == 201
    body = response.get_json()
    assert body["chunks"] >= 1


def test_upload_failure_is_500_without_a_traceback(client, monkeypatch):
    def boom(self, path):
        raise RuntimeError("pypdf exploded")

    monkeypatch.setattr("rag.api.DocumentIngestor.ingest", boom)
    data = {"file": (io.BytesIO(_pdf_bytes()), "book.pdf")}

    response = client.post("/upload", data=data, content_type="multipart/form-data")

    assert response.status_code == 500
    assert response.get_json() == {"error": "failed to process document"}


# --- /query ---

def test_query_before_any_upload_is_404(client):
    response = client.post("/query", json={"question": "what is deep learning?"})

    assert response.status_code == 404
    assert "error" in response.get_json()


def test_query_missing_question_is_400(client):
    response = client.post("/query", json={})

    assert response.status_code == 400


def test_query_blank_question_is_400(client):
    response = client.post("/query", json={"question": "   "})

    assert response.status_code == 400


def test_query_non_json_body_is_400(client):
    response = client.post("/query", data="not json", content_type="text/plain")

    assert response.status_code == 400


def test_query_after_upload_returns_answer_and_pages(client):
    upload = client.post(
        "/upload",
        data={"file": (io.BytesIO(_pdf_bytes()), "book.pdf")},
        content_type="multipart/form-data",
    )
    assert upload.status_code == 201

    response = client.post("/query", json={"question": "what is deep learning?"})

    assert response.status_code == 200
    body = response.get_json()
    assert body["answer"] == "stub answer"
    assert isinstance(body["pages"], list)
    assert body["num_sources"] >= 1


def test_query_failure_is_500(client, monkeypatch):
    client.post(
        "/upload",
        data={"file": (io.BytesIO(_pdf_bytes()), "book.pdf")},
        content_type="multipart/form-data",
    )

    def boom(self, question):
        raise RuntimeError("gemini exploded")

    monkeypatch.setattr("rag.pipeline.RAGPipeline.ask", boom)
    response = client.post("/query", json={"question": "anything"})

    assert response.status_code == 500
    assert response.get_json() == {"error": "failed to answer question"}


def test_upload_too_large_is_413(config, embeddings, monkeypatch):
    from conftest import FakeLLM

    monkeypatch.setattr("rag.api.MAX_UPLOAD_BYTES", 16)
    knowledge_base = KnowledgeBase(config, embeddings=embeddings)
    app = create_app(config=config, knowledge_base=knowledge_base, llm=FakeLLM())
    client = app.test_client()

    data = {"file": (io.BytesIO(_pdf_bytes()), "book.pdf")}
    response = client.post("/upload", data=data, content_type="multipart/form-data")

    assert response.status_code == 413
    assert response.get_json() == {"error": "file too large"}


def test_unexpected_error_is_masked_as_500(client, monkeypatch):
    # An error raised outside the endpoints' own try/except must still be
    # caught by the backstop handler and returned without a traceback.
    def boom(self):
        raise RuntimeError("something deep broke")

    monkeypatch.setattr(
        "rag.knowledge_base.KnowledgeBase.is_populated",
        property(boom),
    )
    response = client.post("/query", json={"question": "anything"})

    assert response.status_code == 500
    assert response.get_json() == {"error": "internal server error"}


def test_default_llm_is_constructed_when_none_injected(config, embeddings, monkeypatch):
    # Cover the production branch that builds a real Gemini client, without
    # a key or network, by swapping the class the factory imports.
    constructed = {}

    class DummyLLM:
        def __init__(self, model):
            constructed["model"] = model

    monkeypatch.setattr("langchain_google_genai.ChatGoogleGenerativeAI", DummyLLM)
    knowledge_base = KnowledgeBase(config, embeddings=embeddings)

    app = create_app(config=config, knowledge_base=knowledge_base)

    assert isinstance(app.llm, DummyLLM)
    assert constructed["model"] == config.llm_model


# --- routing ---

def test_unknown_route_is_404_json(client):
    response = client.get("/nope")

    assert response.status_code == 404
    assert response.is_json


def test_wrong_method_is_405_json(client):
    response = client.get("/query")

    assert response.status_code == 405
    assert response.is_json
