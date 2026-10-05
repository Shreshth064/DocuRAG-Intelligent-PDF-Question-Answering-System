from dataclasses import dataclass


@dataclass(frozen=True)
class RAGConfig:
    """Single source of truth for every tunable in the pipeline."""

    embedding_model: str = "BAAI/bge-m3"
    # The lite model is less congested than gemini-flash-latest and has its
    # own free-tier quota, so it hits fewer 429/503 responses.
    llm_model: str = "gemini-flash-lite-latest"
    # Retries for transient (429/5xx) LLM failures; attempts include the first.
    llm_max_attempts: int = 3
    llm_retry_initial_wait: float = 1.0
    llm_retry_max_wait: float = 10.0
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
