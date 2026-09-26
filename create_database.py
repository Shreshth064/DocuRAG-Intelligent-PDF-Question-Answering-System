from dotenv import load_dotenv

from rag import DocumentIngestor, KnowledgeBase, RAGConfig

PDF_PATH = "document loader/deeplearning.pdf"


def main() -> None:
    load_dotenv()
    config = RAGConfig()

    chunks = DocumentIngestor(config).ingest(PDF_PATH)
    print(f"Split into {len(chunks)} chunks")

    KnowledgeBase(config).build(chunks)
    print(f"Vector database written to {config.persist_directory}")


if __name__ == "__main__":
    main()
