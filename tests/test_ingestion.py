from langchain_core.documents import Document

from rag import DocumentIngestor, RAGConfig


def test_split_honours_chunk_size():
    config = RAGConfig(chunk_size=50, chunk_overlap=10)
    long_document = Document(page_content="word " * 200)

    chunks = DocumentIngestor(config).split([long_document])

    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= 50 for chunk in chunks)


def test_chunks_overlap_so_context_is_not_cut_mid_idea():
    config = RAGConfig(chunk_size=100, chunk_overlap=40)
    text = " ".join(f"token{i}" for i in range(200))

    chunks = DocumentIngestor(config).split([Document(page_content=text)])
    first_words = set(chunks[0].page_content.split())
    second_words = set(chunks[1].page_content.split())

    assert first_words & second_words, "consecutive chunks should share content"


def test_short_document_stays_a_single_chunk():
    config = RAGConfig(chunk_size=1000, chunk_overlap=200)

    chunks = DocumentIngestor(config).split([Document(page_content="short text")])

    assert len(chunks) == 1
    assert chunks[0].page_content == "short text"


def test_split_preserves_metadata():
    config = RAGConfig(chunk_size=50, chunk_overlap=10)
    document = Document(page_content="word " * 100, metadata={"page": 4, "source": "x.pdf"})

    chunks = DocumentIngestor(config).split([document])

    assert all(chunk.metadata["page"] == 4 for chunk in chunks)
    assert all(chunk.metadata["source"] == "x.pdf" for chunk in chunks)


def test_load_reads_every_page_of_a_real_pdf(tiny_pdf):
    pages = DocumentIngestor().load(str(tiny_pdf))

    assert len(pages) == 2
    assert "gradient descent" in pages[0].page_content.lower()
    assert "attention" in pages[1].page_content.lower()


def test_ingest_loads_and_splits_in_one_call(tiny_pdf):
    config = RAGConfig(chunk_size=20, chunk_overlap=5)

    chunks = DocumentIngestor(config).ingest(str(tiny_pdf))

    # Two short pages split at 20 chars must produce more chunks than pages.
    assert len(chunks) > 2
    assert all(len(chunk.page_content) <= 20 for chunk in chunks)
