"""An agentic alternative to the fixed retrieve-then-answer pipeline.

Instead of always running one retrieval and one LLM call, a tool-calling
agent decides which document-scoped tools to call and in what order (search,
cite, summarise, calculate) before answering. The tools only ever read the
loaded document, and the system prompt forbids outside knowledge, so the
agent keeps the pipeline's grounding guarantee.
"""

from dataclasses import dataclass, field

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI

from .config import RAGConfig
from .knowledge_base import KnowledgeBase
from .pipeline import NOT_FOUND_MESSAGE, _as_text
from .tools import DocumentToolkit

# Bounds a confused agent's tool loop; a grounded answer rarely needs more
# than search -> cite -> calculate.
DEFAULT_MAX_ITERATIONS = 6

AGENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI assistant that answers questions about the "
            "loaded document.\n\n"
            "Answer ONLY from what the tools return. They are your sole source "
            "of information: never use prior or background knowledge, and "
            "never guess.\n\n"
            "Use search_documents to find relevant passages, get_citations to "
            "report which pages back your answer, summarize_document for "
            "overview questions, and calculator for any arithmetic on figures "
            "the document states.\n\n"
            "If the tools do not return the information needed to answer, "
            f'say exactly: "{NOT_FOUND_MESSAGE}"',
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


def build_agent(
    llm: BaseChatModel,
    tools: list[BaseTool],
    prompt: ChatPromptTemplate = AGENT_PROMPT,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> AgentExecutor:
    """Wire a tool-calling agent and its executor over the given tools."""
    return AgentExecutor(
        agent=create_tool_calling_agent(llm, tools, prompt),
        tools=tools,
        max_iterations=max_iterations,
        return_intermediate_steps=True,
    )


@dataclass(frozen=True)
class ToolCall:
    tool: str
    input: dict | str
    output: str


@dataclass(frozen=True)
class AgentAnswer:
    text: str
    steps: list[ToolCall] = field(default_factory=list)

    @property
    def tools_used(self) -> list[str]:
        return [step.tool for step in self.steps]


class DocumentAgent:
    """Answers questions by letting the LLM choose and chain document tools."""

    def __init__(self, executor: AgentExecutor):
        self._executor = executor

    @classmethod
    def from_knowledge_base(
        cls,
        knowledge_base: KnowledgeBase,
        llm: BaseChatModel,
        config: RAGConfig | None = None,
    ) -> "DocumentAgent":
        tools = DocumentToolkit.from_knowledge_base(knowledge_base, llm, config).as_tools()
        return cls(build_agent(llm, tools))

    @classmethod
    def from_config(cls, config: RAGConfig | None = None) -> "DocumentAgent":
        config = config or RAGConfig()
        return cls.from_knowledge_base(
            KnowledgeBase(config),
            ChatGoogleGenerativeAI(model=config.llm_model),
            config,
        )

    def ask(self, question: str) -> AgentAnswer:
        result = self._executor.invoke({"input": question})
        steps = [
            ToolCall(tool=action.tool, input=action.tool_input, output=str(observation))
            for action, observation in result["intermediate_steps"]
        ]
        return AgentAnswer(text=_as_text(result["output"]), steps=steps)
