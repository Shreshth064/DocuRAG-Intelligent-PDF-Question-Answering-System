import dataclasses

import pytest

from rag import RAGConfig


def test_search_kwargs_mirror_the_config():
    config = RAGConfig(retriever_k=3, retriever_fetch_k=9, retriever_lambda_mult=0.25)

    assert config.search_kwargs == {"k": 3, "fetch_k": 9, "lambda_mult": 0.25}


def test_defaults_are_sane():
    config = RAGConfig()

    assert config.chunk_overlap < config.chunk_size
    assert config.retriever_k <= config.retriever_fetch_k
    assert 0.0 <= config.retriever_lambda_mult <= 1.0


def test_config_is_immutable():
    config = RAGConfig()

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.llm_model = "something-else"


def test_overriding_one_field_leaves_the_rest_alone():
    default = RAGConfig()
    overridden = RAGConfig(persist_directory="chroma_db")

    assert overridden.persist_directory == "chroma_db"
    assert overridden.embedding_model == default.embedding_model
    assert overridden.llm_model == default.llm_model
    assert overridden.chunk_size == default.chunk_size
