import pytest
from langchain_core.documents import Document

from rag import KnowledgeBase, RAGConfig

pytestmark = pytest.mark.integration


def test_empty_directory_does_not_count_as_populated(config, embeddings):
    assert KnowledgeBase(config, embeddings=embeddings).is_populated is False


def test_build_marks_the_store_populated(config, embeddings, sample_documents):
    knowledge_base = KnowledgeBase(config, embeddings=embeddings)

    knowledge_base.build(sample_documents)

    assert knowledge_base.is_populated


def test_build_returns_self_so_calls_can_chain(config, embeddings, sample_documents):
    knowledge_base = KnowledgeBase(config, embeddings=embeddings)

    assert knowledge_base.build(sample_documents) is knowledge_base


def test_store_is_opened_once_and_reused(config, embeddings):
    knowledge_base = KnowledgeBase(config, embeddings=embeddings)

    assert knowledge_base.store is knowledge_base.store


def test_a_fresh_instance_sees_persisted_vectors(config, embeddings, sample_documents):
    KnowledgeBase(config, embeddings=embeddings).build(sample_documents)

    reopened = KnowledgeBase(config, embeddings=embeddings)

    assert reopened.is_populated


def test_fingerprint_is_stable_for_an_unchanged_store(config, embeddings, sample_documents):
    knowledge_base = KnowledgeBase(config, embeddings=embeddings)
    knowledge_base.build(sample_documents)

    assert knowledge_base.fingerprint == knowledge_base.fingerprint


def test_fingerprint_survives_reopening_the_store(config, embeddings, sample_documents):
    KnowledgeBase(config, embeddings=embeddings).build(sample_documents)

    first = KnowledgeBase(config, embeddings=embeddings).fingerprint
    second = KnowledgeBase(config, embeddings=embeddings).fingerprint

    assert first == second


def test_fingerprint_changes_when_the_corpus_changes(config, embeddings, sample_documents):
    knowledge_base = KnowledgeBase(config, embeddings=embeddings)
    knowledge_base.build(sample_documents)
    before = knowledge_base.fingerprint

    knowledge_base.build(sample_documents[:1])

    assert knowledge_base.fingerprint != before


def test_retriever_returns_at_most_k_documents(tmp_path, embeddings, sample_documents):
    config = RAGConfig(
        persist_directory=str(tmp_path / "store"),
        retriever_k=2,
        retriever_fetch_k=3,
    )
    KnowledgeBase(config, embeddings=embeddings).build(sample_documents)

    hits = KnowledgeBase(config, embeddings=embeddings).as_retriever().invoke(
        "how are gradients computed?"
    )

    assert len(hits) == 2
    assert all(isinstance(hit, Document) for hit in hits)


def test_retriever_k_follows_the_config(tmp_path, embeddings, sample_documents):
    config = RAGConfig(
        persist_directory=str(tmp_path / "store"),
        retriever_k=1,
        retriever_fetch_k=3,
    )
    KnowledgeBase(config, embeddings=embeddings).build(sample_documents)

    hits = KnowledgeBase(config, embeddings=embeddings).as_retriever().invoke("gradients")

    assert len(hits) == 1


def test_retriever_k_can_be_overridden_per_call(tmp_path, embeddings, sample_documents):
    # fetch_k is below the override, so it must widen to keep MMR satisfiable.
    config = RAGConfig(
        persist_directory=str(tmp_path / "store"),
        retriever_k=1,
        retriever_fetch_k=2,
    )
    knowledge_base = KnowledgeBase(config, embeddings=embeddings).build(sample_documents)

    hits = knowledge_base.as_retriever(k=3).invoke("gradients")

    assert len(hits) == 3
    assert len(knowledge_base.as_retriever().invoke("gradients")) == 1
