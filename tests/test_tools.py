import pytest
from conftest import FakeLLM, FakeRetriever
from langchain_core.documents import Document

from rag import DocumentToolkit, KnowledgeBase, RAGConfig
from rag.pipeline import NOT_FOUND_MESSAGE
from rag.tools import NO_RESULTS_MESSAGE, UnsafeExpressionError, safe_eval


def _toolkit(documents, reply="stub summary", **kwargs):
    llm = FakeLLM(reply)
    retriever = FakeRetriever(documents)
    return DocumentToolkit(retriever=retriever, llm=llm, **kwargs), llm, retriever


# --- calculator: safe arithmetic ---

@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 3", 5),
        ("10 - 4 * 2", 2),
        ("(120 - 80) / 80", 0.5),
        ("2 ** 10", 1024),
        ("2 ** -1", 0.5),
        ("1.5 * 4", 6.0),
        ("-2 ** 2", -4),
        ("+3", 3),
        ("-(3 - 5)", 2),
        ("2.0 ** 3", 8.0),
        ("  7 / 2  ", 3.5),
    ],
)
def test_safe_eval_computes_arithmetic(expression, expected):
    assert safe_eval(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('ls')",  # call
        "x + 1",  # name
        "(1).real",  # attribute access
        "[1, 2][0]",  # subscript / list
        "'a' * 3",  # non-numeric constant
        "True + 1",  # bool is not a figure
        "7 % 3",  # operator outside the whitelist
        "not 1",  # unary operator outside the whitelist
        "1 if 1 else 2",  # conditional expression
        "lambda: 1",  # lambda
    ],
)
def test_safe_eval_rejects_anything_but_arithmetic(expression):
    with pytest.raises(UnsafeExpressionError):
        safe_eval(expression)


def test_safe_eval_rejects_syntax_errors():
    with pytest.raises(UnsafeExpressionError, match="not a valid expression"):
        safe_eval("1 +")


def test_safe_eval_rejects_huge_exponents():
    with pytest.raises(UnsafeExpressionError, match="exponent too large"):
        safe_eval("9 ** 9 ** 9")


def test_safe_eval_rejects_power_towers_that_would_explode():
    # Each exponent is within the cap, but the result would be enormous.
    with pytest.raises(UnsafeExpressionError, match="result too large"):
        safe_eval("(10 ** 90) ** 60")


def test_safe_eval_rejects_complex_results():
    with pytest.raises(UnsafeExpressionError, match="not a real number"):
        safe_eval("(-8) ** 0.5")


def test_unsafe_expression_error_is_a_value_error():
    assert issubclass(UnsafeExpressionError, ValueError)


def test_calculator_returns_the_result_as_text():
    assert DocumentToolkit.calculator("6 * 7") == "42"


def _tool(toolkit, name):
    return {tool.name: tool for tool in toolkit.as_tools()}[name]


def test_calculator_tool_reports_unsafe_input_back_to_the_agent():
    toolkit, _, _ = _toolkit([])

    result = _tool(toolkit, "calculator").invoke({"expression": "open('x')"})

    assert result.startswith("calculator error: disallowed syntax: Call")


def test_calculator_tool_reports_arithmetic_errors_back_to_the_agent():
    toolkit, _, _ = _toolkit([])

    result = _tool(toolkit, "calculator").invoke({"expression": "1 / 0"})

    assert result == "calculator error: division by zero"


# --- search_documents ---

def test_search_returns_only_the_retrieved_text(sample_documents):
    toolkit, _, retriever = _toolkit(sample_documents)

    result = toolkit.search_documents("how are gradients computed?")

    assert retriever.last_question == "how are gradients computed?"
    assert result == (
        "[1] Gradient descent optimises the loss.\n\n"
        "[2] Backpropagation computes gradients.\n\n"
        "[3] Transformers use self attention."
    )


def test_search_with_no_hits_says_so(sample_documents):
    toolkit, _, _ = _toolkit([])

    assert toolkit.search_documents("anything") == NO_RESULTS_MESSAGE


# --- get_citations ---

def test_citations_list_pages_sources_and_chunk_ids():
    documents = [
        Document(
            page_content="text",
            metadata={"page": 4, "source": "paper.pdf"},
            id="abc123",
        ),
        Document(page_content="more", metadata={"page": 9}),
        Document(page_content="bare"),
    ]
    toolkit, _, retriever = _toolkit(documents)

    result = toolkit.get_citations("attention")

    assert retriever.last_question == "attention"
    assert result.splitlines() == [
        "[1] page 4, source paper.pdf, chunk abc123",
        "[2] page 9",
        "[3] no source metadata",
    ]


def test_citations_do_not_leak_chunk_text(sample_documents):
    toolkit, _, _ = _toolkit(sample_documents)

    result = toolkit.get_citations("gradients")

    assert "Gradient descent" not in result
    assert "page 1" in result


def test_citations_with_no_hits_say_so():
    toolkit, _, _ = _toolkit([])

    assert toolkit.get_citations("anything") == NO_RESULTS_MESSAGE


# --- summarize_document ---

def test_summary_is_the_llm_reply(sample_documents):
    toolkit, _, _ = _toolkit(sample_documents, reply="a grounded summary")

    assert toolkit.summarize_document("training") == "a grounded summary"


def test_summary_prompt_contains_only_the_retrieved_chunks(sample_documents):
    toolkit, llm, retriever = _toolkit(sample_documents)

    toolkit.summarize_document("training")

    prompt = llm.rendered_prompt
    assert retriever.last_question == "training"
    for doc in sample_documents:
        assert doc.page_content in prompt
    assert "training" in prompt
    assert "ONLY the provided excerpts" in prompt
    assert NOT_FOUND_MESSAGE in prompt


def test_summary_without_context_abstains_without_calling_the_llm():
    toolkit, llm, _ = _toolkit([])

    assert toolkit.summarize_document("anything") == NOT_FOUND_MESSAGE
    assert llm.call_count == 0


def test_summary_flattens_structured_llm_content(sample_documents):
    toolkit, _, _ = _toolkit(
        sample_documents, reply=[{"type": "text", "text": "flat"}, " summary"]
    )

    assert toolkit.summarize_document("training") == "flat summary"


def test_summary_uses_its_own_broader_retriever_when_given(sample_documents):
    broad = FakeRetriever(sample_documents)
    toolkit, llm, narrow = _toolkit(sample_documents[:1], summary_retriever=broad)

    toolkit.summarize_document("everything")

    assert broad.last_question == "everything"
    assert narrow.last_question is None
    assert "Transformers use self attention." in llm.rendered_prompt


# --- LangChain tool wrappers ---

def test_as_tools_exposes_the_four_document_scoped_tools():
    toolkit, _, _ = _toolkit([])

    tools = toolkit.as_tools()

    assert [tool.name for tool in tools] == [
        "search_documents",
        "get_citations",
        "summarize_document",
        "calculator",
    ]
    assert all(tool.description for tool in tools)


def test_tools_are_invocable_through_langchain(sample_documents):
    toolkit, _, _ = _toolkit(sample_documents, reply="summary")

    assert "Backpropagation" in _tool(toolkit, "search_documents").invoke({"query": "q"})
    assert "page 2" in _tool(toolkit, "get_citations").invoke({"query": "q"})
    assert _tool(toolkit, "summarize_document").invoke({"topic": "t"}) == "summary"
    assert _tool(toolkit, "calculator").invoke({"expression": "2 + 2"}) == "4"


# --- factory ---

def test_from_knowledge_base_sweeps_more_chunks_for_summaries(
    tmp_path, embeddings, sample_documents
):
    config = RAGConfig(
        persist_directory=str(tmp_path / "store"),
        retriever_k=1,
        retriever_fetch_k=3,
        summary_k=3,
    )
    knowledge_base = KnowledgeBase(config, embeddings=embeddings).build(sample_documents)
    llm = FakeLLM("summary")

    toolkit = DocumentToolkit.from_knowledge_base(knowledge_base, llm, config)

    assert toolkit.search_documents("gradients").count("[") == 1
    toolkit.summarize_document("gradients")
    for doc in sample_documents:
        assert doc.page_content in llm.rendered_prompt


def test_from_knowledge_base_defaults_the_config(config, embeddings, sample_documents):
    knowledge_base = KnowledgeBase(config, embeddings=embeddings).build(sample_documents)

    toolkit = DocumentToolkit.from_knowledge_base(knowledge_base, FakeLLM())

    assert isinstance(toolkit, DocumentToolkit)
