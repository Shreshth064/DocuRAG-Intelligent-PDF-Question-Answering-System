import dataclasses

import pytest
from langchain_core.documents import Document

from rag import Answer, RAGPipeline
from rag.pipeline import NOT_FOUND_MESSAGE, _as_text


def test_ask_returns_the_llm_reply(make_pipeline):
    pipeline, _, _ = make_pipeline([Document(page_content="chunk")], reply="42")

    assert pipeline.ask("what is the answer?").text == "42"


def test_retrieved_chunks_reach_the_prompt(make_pipeline):
    docs = [Document(page_content="alpha"), Document(page_content="beta")]
    pipeline, llm, _ = make_pipeline(docs)

    pipeline.ask("question?")

    assert "alpha" in llm.rendered_prompt
    assert "beta" in llm.rendered_prompt
    assert "question?" in llm.rendered_prompt


def test_prompt_tells_the_model_to_refuse_outside_the_context(make_pipeline):
    pipeline, llm, _ = make_pipeline([Document(page_content="alpha")])

    pipeline.ask("question?")

    assert NOT_FOUND_MESSAGE in llm.rendered_prompt
    assert "ONLY the provided context" in llm.rendered_prompt


def test_the_question_is_forwarded_to_the_retriever(make_pipeline):
    pipeline, _, retriever = make_pipeline([Document(page_content="alpha")])

    pipeline.ask("what is backpropagation?")

    assert retriever.last_question == "what is backpropagation?"


def test_answer_carries_its_sources(make_pipeline):
    docs = [Document(page_content="alpha"), Document(page_content="beta")]
    pipeline, _, _ = make_pipeline(docs)

    assert pipeline.ask("question?").sources == docs


def test_chunks_are_separated_in_the_context(make_pipeline):
    docs = [Document(page_content="alpha"), Document(page_content="beta")]
    pipeline, llm, _ = make_pipeline(docs)

    pipeline.ask("question?")

    assert "alpha\n\nbeta" in llm.rendered_prompt


def test_empty_retrieval_still_produces_an_answer(make_pipeline):
    pipeline, llm, _ = make_pipeline([], reply=NOT_FOUND_MESSAGE)

    answer = pipeline.ask("something unrelated")

    assert answer.text == NOT_FOUND_MESSAGE
    assert answer.sources == []
    assert answer.pages == []


def test_the_llm_is_called_once_per_question(make_pipeline):
    pipeline, llm, _ = make_pipeline([Document(page_content="alpha")])

    pipeline.ask("first?")
    pipeline.ask("second?")

    assert llm.call_count == 2


def test_retrieve_exposes_documents_without_calling_the_llm(make_pipeline):
    docs = [Document(page_content="alpha")]
    pipeline, llm, _ = make_pipeline(docs)

    assert pipeline.retrieve("question?") == docs
    assert llm.call_count == 0


def test_answer_pages_are_sorted_and_deduped():
    answer = Answer(
        text="irrelevant",
        sources=[
            Document(page_content="a", metadata={"page": 7}),
            Document(page_content="b", metadata={"page": 2}),
            Document(page_content="c", metadata={"page": 7}),
            Document(page_content="d", metadata={}),
        ],
    )

    assert answer.pages == [2, 7]


def test_answer_without_sources_has_no_pages():
    assert Answer(text="hello").pages == []


def test_answer_is_immutable():
    answer = Answer(text="hello")

    with pytest.raises(dataclasses.FrozenInstanceError):
        answer.text = "changed"


def test_a_custom_prompt_can_be_injected(make_pipeline):
    from langchain_core.prompts import ChatPromptTemplate

    custom = ChatPromptTemplate.from_messages(
        [("human", "CUSTOM {context} :: {question}")]
    )
    pipeline, llm, _ = make_pipeline([Document(page_content="alpha")], prompt=custom)

    pipeline.ask("question?")

    assert llm.rendered_prompt.startswith("Human: CUSTOM alpha :: question?")


# --- structured-content flattening ---

def test_as_text_passes_through_a_plain_string():
    assert _as_text("just a string") == "just a string"


def test_as_text_joins_structured_content_blocks():
    # Mirrors Gemini's structured output: text parts kept, everything else
    # (a string part, a text dict, an image dict, a non-string/dict) folded
    # into the two branches that keep text and the two that drop it.
    content = [
        "lead ",
        {"type": "text", "text": "answer body"},
        {"type": "image", "url": "http://x"},
        {"type": "thinking", "signature": "abc"},
        42,
    ]

    assert _as_text(content) == "lead answer body"


def test_ask_flattens_structured_llm_content(make_pipeline):
    from langchain_core.documents import Document

    pipeline, llm, _ = make_pipeline([Document(page_content="alpha")])
    # Make the fake LLM return blocks the way Gemini does.
    llm.reply = [{"type": "text", "text": "flattened"}]

    assert pipeline.ask("question?").text == "flattened"


def test_from_config_builds_a_usable_pipeline(config, monkeypatch):
    """The factory must wire a real retriever and LLM without being called."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-not-used")

    pipeline = RAGPipeline.from_config(config)

    assert isinstance(pipeline, RAGPipeline)
