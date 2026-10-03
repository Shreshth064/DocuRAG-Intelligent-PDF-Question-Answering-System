import hashlib

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings

from .config import RAGConfig


class KnowledgeBase:
    """Owns the embedding model and the Chroma vector store behind it."""

    def __init__(self, config: RAGConfig | None = None, embeddings: Embeddings | None = None):
        self._config = config or RAGConfig()
        # Hugging Face Inference API, so the model is never downloaded locally.
        self._embeddings = embeddings or HuggingFaceEndpointEmbeddings(
            model=self._config.embedding_model
        )
        self._store: Chroma | None = None

    @property
    def is_populated(self) -> bool:
        """True only if the store holds documents; an empty directory is not enough."""
        return bool(self.store.get(limit=1)["ids"])

    @property
    def fingerprint(self) -> str:
        """A stable identifier for the current corpus.

        Derived from the stored document ids, so it changes whenever the set
        of documents changes and a rebuilt store can never serve a previous
        store's cached answers. Hashing the ids (rather than an in-process
        counter) keeps the identity stable across a restart, so a warm cache
        survives one.
        """
        ids = self.store.get(include=[])["ids"]
        return hashlib.sha256(",".join(sorted(ids)).encode()).hexdigest()

    @property
    def store(self) -> Chroma:
        if self._store is None:
            self._store = Chroma(
                persist_directory=self._config.persist_directory,
                embedding_function=self._embeddings,
            )
        return self._store

    def build(self, chunks: list[Document]) -> "KnowledgeBase":
        self._store = Chroma.from_documents(
            documents=chunks,
            embedding=self._embeddings,
            persist_directory=self._config.persist_directory,
        )
        return self

    def as_retriever(self, k: int | None = None):
        """An MMR retriever over the store.

        k overrides the configured number of chunks returned, for callers
        that need a broader sweep (e.g. summarisation). The candidate pool
        is widened to match so MMR always has at least k to choose from.
        """
        search_kwargs = self._config.search_kwargs
        if k is not None:
            search_kwargs |= {"k": k, "fetch_k": max(k, search_kwargs["fetch_k"])}
        return self.store.as_retriever(
            search_type="mmr",
            search_kwargs=search_kwargs,
        )
