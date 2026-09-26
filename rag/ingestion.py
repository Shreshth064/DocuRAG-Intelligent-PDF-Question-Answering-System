from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import RAGConfig


class DocumentIngestor:
    """Turns a PDF on disk into retrieval-sized chunks."""

    def __init__(self, config: RAGConfig | None = None):
        self._config = config or RAGConfig()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._config.chunk_size,
            chunk_overlap=self._config.chunk_overlap,
        )

    def load(self, pdf_path: str) -> list[Document]:
        return PyPDFLoader(pdf_path).load()

    def split(self, documents: list[Document]) -> list[Document]:
        return self._splitter.split_documents(documents)

    def ingest(self, pdf_path: str) -> list[Document]:
        return self.split(self.load(pdf_path))
