"""A thin Flask adapter over the RAG engine.

This module owns HTTP concerns only. All retrieval logic stays in the
other rag/ modules; the API constructs the engine once at startup and
each request calls straight into it.
"""

import contextlib
import os
import tempfile
import threading

from flask import Flask, current_app, jsonify, request
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename

from .agent import DocumentAgent
from .cache import QueryCache
from .config import RAGConfig
from .ingestion import DocumentIngestor
from .knowledge_base import KnowledgeBase
from .pipeline import RAGPipeline

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
_PDF_MAGIC = b"%PDF-"


class ApiError(Exception):
    """An error that maps directly onto an HTTP status and a JSON body."""

    status_code = 500
    message = "internal server error"

    def __init__(self, message: str | None = None):
        super().__init__(message or self.message)
        if message is not None:
            self.message = message


class BadRequest(ApiError):
    status_code = 400
    message = "bad request"


class UnsupportedMedia(ApiError):
    status_code = 415
    message = "unsupported media type"


class StoreNotReady(ApiError):
    status_code = 404
    message = "no document has been uploaded yet"


def _validate_pdf(file_storage) -> None:
    """Reject anything that is not really a PDF, sniffing bytes not headers.

    The client-supplied filename and content type are advisory; the magic
    number is the only trustworthy signal, so we check that and rewind.
    """
    if file_storage is None or not file_storage.filename:
        raise BadRequest("no file provided")

    header = file_storage.stream.read(len(_PDF_MAGIC))
    file_storage.stream.seek(0)
    if header != _PDF_MAGIC:
        raise UnsupportedMedia("not a PDF")


def _rebuild_pipeline(app: Flask) -> None:
    """Point the pipeline at the current store after a rebuild.

    build() swaps in a fresh Chroma object, so a retriever captured before
    the upload still wraps the old one. Only the cheap pipeline wrapper is
    recreated here; the embedding client and the LLM are reused.
    """
    app.rag_pipeline = RAGPipeline(
        retriever=app.knowledge_base.as_retriever(),
        llm=app.llm,
    )


def create_app(config=None, knowledge_base=None, llm=None, query_cache=None) -> Flask:
    """Application factory.

    In production, pass nothing and the real embedding client, store and
    LLM are built once. Tests inject a KnowledgeBase backed by fake
    embeddings, a fake LLM, and a cache over a fake Redis, so the whole API
    runs offline.
    """
    app = Flask(__name__)
    config = config or RAGConfig()

    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    app.rag_config = config
    app.knowledge_base = knowledge_base or KnowledgeBase(config)
    # from_env never raises: an unreachable Redis yields a disabled cache, so
    # a cache outage cannot crash startup.
    app.query_cache = query_cache if query_cache is not None else QueryCache.from_env()
    if llm is not None:
        app.llm = llm
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI

        app.llm = ChatGoogleGenerativeAI(model=config.llm_model)
    app.build_lock = threading.Lock()
    _rebuild_pipeline(app)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"}), 200

    @app.post("/upload")
    def upload():
        file_storage = request.files.get("file")
        _validate_pdf(file_storage)

        safe_name = secure_filename(file_storage.filename) or "upload.pdf"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp_path = tmp.name
        try:
            file_storage.save(tmp)
            tmp.close()

            chunks = DocumentIngestor(current_app.rag_config).ingest(tmp_path)
            with current_app.build_lock:
                current_app.knowledge_base.build(chunks)
                _rebuild_pipeline(current_app)
        except Exception as exc:
            current_app.logger.exception("ingestion failed for %s", safe_name)
            raise ApiError("failed to process document") from exc
        finally:
            tmp.close()
            with contextlib.suppress(FileNotFoundError):
                os.remove(tmp_path)

        return jsonify({"chunks": len(chunks)}), 201

    def _question_for_populated_store() -> str:
        payload = request.get_json(silent=True) or {}
        question = payload.get("question")
        if not isinstance(question, str) or not question.strip():
            raise BadRequest("question is required")

        if not current_app.knowledge_base.is_populated:
            raise StoreNotReady()
        return question

    @app.post("/query")
    def query():
        question = _question_for_populated_store()

        cache = current_app.query_cache
        store_id = current_app.knowledge_base.fingerprint

        cached = cache.get(question, store_id)
        if cached is not None:
            current_app.logger.info("cache hit; skipping retrieval and LLM")
            return jsonify(cached), 200

        try:
            answer = current_app.rag_pipeline.ask(question)
        except Exception as exc:
            current_app.logger.exception("query failed")
            raise ApiError("failed to answer question") from exc

        body = {
            "answer": answer.text,
            "pages": answer.pages,
            "num_sources": len(answer.sources),
        }
        cache.set(question, store_id, body)
        current_app.logger.info("cache miss; computed and stored answer")
        return jsonify(body), 200

    @app.post("/agent")
    def agent():
        """The agentic path: the LLM chooses and chains document-scoped tools.

        Built per request (binding tools is cheap) so the agent always wraps
        the current store, even straight after a re-upload. Not cached: the
        tool trace is part of the response and is worth seeing fresh.
        """
        question = _question_for_populated_store()

        try:
            with current_app.build_lock:
                document_agent = DocumentAgent.from_knowledge_base(
                    current_app.knowledge_base,
                    current_app.llm,
                    current_app.rag_config,
                )
            answer = document_agent.ask(question)
        except Exception as exc:
            current_app.logger.exception("agent query failed")
            raise ApiError("failed to answer question") from exc

        return jsonify(
            {
                "answer": answer.text,
                "tools_used": answer.tools_used,
                "steps": [
                    {"tool": step.tool, "input": step.input} for step in answer.steps
                ],
            }
        ), 200

    @app.errorhandler(ApiError)
    def handle_api_error(exc: ApiError):
        return jsonify({"error": exc.message}), exc.status_code

    @app.errorhandler(413)
    def handle_too_large(_exc):
        return jsonify({"error": "file too large"}), 413

    @app.errorhandler(HTTPException)
    def handle_http_exception(exc: HTTPException):
        return jsonify({"error": exc.name.lower()}), exc.code

    @app.errorhandler(Exception)
    def handle_unexpected(exc: Exception):
        app.logger.exception("unhandled error")
        return jsonify({"error": "internal server error"}), 500

    return app


def main() -> None:  # pragma: no cover - dev-server entrypoint
    import logging

    from dotenv import load_dotenv

    from .environment import require

    # Flask's app.logger defaults to WARNING, which hides the cache hit/miss
    # INFO lines. Configure logging here (only for the real server, never the
    # test process) so cache behaviour is observable in the container logs.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    load_dotenv()
    require("GOOGLE_API_KEY", "HUGGINGFACEHUB_API_TOKEN")
    create_app().run(host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
