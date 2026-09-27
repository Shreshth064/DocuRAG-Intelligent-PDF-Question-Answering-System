from dataclasses import dataclass


@dataclass(frozen=True)
class RAGConfig:
    """Single source of truth for every tunable in the pipeline."""

    embedding_model: str = "BAAI/bge-m3"
    llm_model: str = "gemini-2.5-flash"
    persist_directory: str = "ChromaDB"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retriever_k: int = 4
    retriever_fetch_k: int = 10
    retriever_lambda_mult: float = 0.5

    @property
    def search_kwargs(self) -> dict:
        return {
            "k": self.retriever_k,
            "fetch_k": self.retriever_fetch_k,
            "lambda_mult": self.retriever_lambda_mult,
        }
