import dataclasses

import pytest
from conftest import FakeLLM, FakeRetriever, ScriptedChatModel, tool_call
from langchain_core.messages import AIMessage

from rag import AgentAnswer, DocumentAgent, DocumentToolkit
from rag.agent import AGENT_PROMPT, ToolCall, build_agent
from rag.pipeline import NOT_FOUND_MESSAGE


def _agent(script, documents=(), summary_reply="stub summary"):
    """A real AgentExecutor over a scripted chat model and fake-backed tools."""
    llm = ScriptedChatModel(script=list(script))
    retriever = FakeRetriever(list(documents))
    tools = DocumentToolkit(retriever=retriever, llm=FakeLLM(summary_reply)).as_tools()
    return DocumentAgent(build_agent(llm, tools)), llm, retriever


def test_agent_runs_the_tool_the_llm_chose_and_returns_its_answer():
    agent, _, _ = _agent(
        [
            tool_call("calculator", expression="(120 - 80) / 80"),
            AIMessage(content="Revenue grew by 50%."),
        ]
    )

    answer = agent.ask("By what fraction did revenue grow?")

    assert answer.text == "Revenue grew by 50%."
    assert answer.tools_used == ["calculator"]
    assert answer.steps == [
        ToolCall(tool="calculator", input={"expression": "(120 - 80) / 80"}, output="0.5")
    ]


def test_agent_feeds_the_tool_result_back_to_the_llm(sample_documents):
    agent, llm, retriever = _agent(
        [
            tool_call("search_documents", query="gradients"),
            AIMessage(content="Backpropagation computes gradients."),
        ],
        documents=sample_documents,
    )

    answer = agent.ask("How are gradients computed?")

    assert retriever.last_question == "gradients"
    assert answer.tools_used == ["search_documents"]
    assert "Backpropagation computes gradients." in answer.steps[0].output
    # The second LLM turn must see the retrieved chunk as the tool result.
    second_turn = "\n".join(str(message.content) for message in llm.prompts[1])
    assert "Backpropagation computes gradients." in second_turn


def test_agent_can_chain_several_tools(sample_documents):
    agent, _, _ = _agent(
        [
            tool_call("search_documents", call_id="c1", query="attention"),
            tool_call("get_citations", call_id="c2", query="attention"),
            AIMessage(content="Transformers use self attention (page 3)."),
        ],
        documents=sample_documents,
    )

    answer = agent.ask("What do transformers use?")

    assert answer.tools_used == ["search_documents", "get_citations"]
    assert "page 3" in answer.steps[1].output


def test_agent_can_summarise(sample_documents):
    agent, _, _ = _agent(
        [tool_call("summarize_document", topic="training"), AIMessage(content="done")],
        documents=sample_documents,
        summary_reply="It covers gradient descent.",
    )

    answer = agent.ask("Summarise the training section")

    assert answer.steps[0].output == "It covers gradient descent."


def test_tool_errors_are_returned_to_the_agent_not_raised():
    agent, llm, _ = _agent(
        [
            tool_call("calculator", expression="import os"),
            AIMessage(content=NOT_FOUND_MESSAGE),
        ]
    )

    answer = agent.ask("compute")

    assert answer.steps[0].output.startswith("calculator error:")
    assert len(llm.prompts) == 2


def test_agent_can_abstain_without_calling_tools():
    agent, _, _ = _agent([AIMessage(content=NOT_FOUND_MESSAGE)])

    answer = agent.ask("What is the capital of France?")

    assert answer.text == NOT_FOUND_MESSAGE
    assert answer.steps == []
    assert answer.tools_used == []


def test_agent_flattens_structured_llm_content():
    agent, _, _ = _agent([AIMessage(content=[{"type": "text", "text": "flat"}])])

    assert agent.ask("q").text == "flat"


def test_all_four_tools_are_bound_to_the_llm():
    _, llm, _ = _agent([])

    assert llm.bound_tools == [
        "search_documents",
        "get_citations",
        "summarize_document",
        "calculator",
    ]


def test_system_prompt_restricts_answers_to_the_document():
    system = AGENT_PROMPT.messages[0].prompt.template

    assert "ONLY from what the tools return" in system
    assert "never use prior or background knowledge" in system
    assert NOT_FOUND_MESSAGE in system


def test_build_agent_bounds_the_tool_loop():
    _, llm, _ = _agent([])

    executor = build_agent(llm, [], max_iterations=2)

    assert executor.max_iterations == 2
    assert executor.return_intermediate_steps is True


def test_agent_answer_is_immutable():
    answer = AgentAnswer(text="x")

    with pytest.raises(dataclasses.FrozenInstanceError):
        answer.text = "y"


def test_from_knowledge_base_wires_the_store(config, embeddings, sample_documents):
    from rag import KnowledgeBase

    knowledge_base = KnowledgeBase(config, embeddings=embeddings).build(sample_documents)
    llm = ScriptedChatModel(
        script=[tool_call("search_documents", query="attention"), AIMessage(content="ok")]
    )

    answer = DocumentAgent.from_knowledge_base(knowledge_base, llm, config).ask("q")

    assert "Transformers use self attention." in answer.steps[0].output


def test_from_config_builds_a_usable_agent(config, monkeypatch):
    """The factory must wire the real store, LLM and tools without being called."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-not-used")

    agent = DocumentAgent.from_config(config)

    assert isinstance(agent, DocumentAgent)
