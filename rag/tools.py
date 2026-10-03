"""Document-scoped tools for the agentic path.

Every tool here reads only from the loaded document(s): a retriever over the
vector store, plus an LLM that is shown nothing but retrieved chunks. There
is deliberately no web search or other outside-knowledge tool, so an agent
that chains these tools keeps the same grounding guarantee as the fixed
retrieve-then-answer pipeline.
"""

import ast
import operator

from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool, StructuredTool, ToolException

from .config import RAGConfig
from .knowledge_base import KnowledgeBase
from .pipeline import NOT_FOUND_MESSAGE, _as_text

NO_RESULTS_MESSAGE = "No relevant passages were found in the document."

# Bounds ** so an input like 9**9**9 (or a nested tower of smaller powers)
# cannot pin the CPU or exhaust memory computing a number no document figure
# would ever need. Float overflow is caught separately as an ArithmeticError.
MAX_EXPONENT = 100
MAX_RESULT_BITS = 4096


class UnsafeExpressionError(ValueError):
    """Raised when a calculator expression contains anything but arithmetic."""


_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _check_power(base: int | float, exponent: int | float) -> None:
    if abs(exponent) > MAX_EXPONENT:
        raise UnsafeExpressionError(f"exponent too large (max {MAX_EXPONENT})")
    if isinstance(base, int) and base.bit_length() * abs(exponent) > MAX_RESULT_BITS:
        raise UnsafeExpressionError("result too large")


def _evaluate(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant):
        # bool is an int subclass; True/False are not figures from a document.
        if type(node.value) in (int, float):
            return node.value
        raise UnsafeExpressionError(f"unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow):
            _check_power(left, right)
        return _BINARY_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_evaluate(node.operand))
    raise UnsafeExpressionError(f"disallowed syntax: {type(node).__name__}")


def safe_eval(expression: str) -> int | float:
    """Evaluate plain arithmetic without eval().

    The expression is parsed to an AST and walked against a whitelist:
    int/float literals, + - * / **, unary +/- and parentheses (which the
    parser folds into the tree). Names, calls, attribute access, subscripts
    and everything else raise UnsafeExpressionError.
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpressionError(f"not a valid expression: {expression!r}") from exc
    result = _evaluate(tree.body)
    if isinstance(result, complex):
        # e.g. (-8) ** 0.5; not a figure anyone means to compute.
        raise UnsafeExpressionError("result is not a real number")
    return result


def _describe_source(index: int, doc: Document) -> str:
    """One citation line: the chunk's position plus whatever provenance it has."""
    parts = []
    if "page" in doc.metadata:
        parts.append(f"page {doc.metadata['page']}")
    if "source" in doc.metadata:
        parts.append(f"source {doc.metadata['source']}")
    if doc.id:
        parts.append(f"chunk {doc.id}")
    return f"[{index}] " + (", ".join(parts) or "no source metadata")


SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You summarise documents.\n\n"
            "Use ONLY the provided excerpts. Do not add facts, figures or "
            "background knowledge that are not stated in them.\n\n"
            "If the excerpts do not cover the topic, "
            f'say: "{NOT_FOUND_MESSAGE}"',
        ),
        (
            "human",
            "Excerpts:\n{context}\n\nSummarise what the document says about:\n{topic}",
        ),
    ]
)


class DocumentToolkit:
    """The agent's tools, composed from a retriever and an LLM.

    The retriever and LLM are injected, so tests can drive every tool over
    fakes. summary_retriever lets summarisation sweep more chunks than a
    lookup; it defaults to the main retriever.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: BaseLanguageModel,
        summary_retriever: BaseRetriever | None = None,
        summary_prompt: ChatPromptTemplate = SUMMARY_PROMPT,
    ):
        self._retriever = retriever
        self._llm = llm
        self._summary_retriever = summary_retriever or retriever
        self._summary_prompt = summary_prompt

    @classmethod
    def from_knowledge_base(
        cls,
        knowledge_base: KnowledgeBase,
        llm: BaseLanguageModel,
        config: RAGConfig | None = None,
    ) -> "DocumentToolkit":
        config = config or RAGConfig()
        return cls(
            retriever=knowledge_base.as_retriever(),
            llm=llm,
            summary_retriever=knowledge_base.as_retriever(k=config.summary_k),
        )

    def search_documents(self, query: str) -> str:
        docs = self._retriever.invoke(query)
        if not docs:
            return NO_RESULTS_MESSAGE
        return "\n\n".join(
            f"[{index}] {doc.page_content}" for index, doc in enumerate(docs, start=1)
        )

    def get_citations(self, query: str) -> str:
        docs = self._retriever.invoke(query)
        if not docs:
            return NO_RESULTS_MESSAGE
        return "\n".join(
            _describe_source(index, doc) for index, doc in enumerate(docs, start=1)
        )

    def summarize_document(self, topic: str) -> str:
        docs = self._summary_retriever.invoke(topic)
        if not docs:
            # Nothing to ground a summary in, so don't give the LLM the chance
            # to improvise one.
            return NOT_FOUND_MESSAGE
        context = "\n\n".join(doc.page_content for doc in docs)
        response = self._llm.invoke(
            self._summary_prompt.invoke({"context": context, "topic": topic})
        )
        return _as_text(response.content)

    @staticmethod
    def calculator(expression: str) -> str:
        try:
            return str(safe_eval(expression))
        except (ValueError, ArithmeticError) as exc:
            # ToolException is reported back to the agent as the tool's
            # result, so it can correct the expression instead of crashing.
            raise ToolException(f"calculator error: {exc}") from exc

    def as_tools(self) -> list[BaseTool]:
        """The four tools as LangChain StructuredTools, ready to bind to an agent."""
        specs = [
            (
                self.search_documents,
                "search_documents",
                "Search the loaded document for passages relevant to a query. "
                "Returns the matching text chunks. Use this first to find facts.",
            ),
            (
                self.get_citations,
                "get_citations",
                "Return the provenance (page numbers, source file, chunk ids) "
                "of the passages that back a query. Use this to cite sources.",
            ),
            (
                self.summarize_document,
                "summarize_document",
                "Produce a summary of what the loaded document says about a "
                "topic, grounded only in retrieved passages.",
            ),
            (
                self.calculator,
                "calculator",
                "Evaluate an arithmetic expression (numbers, + - * / **, "
                "parentheses) on figures found in the document, e.g. '(120 - 80) / 80'.",
            ),
        ]
        return [
            StructuredTool.from_function(
                func=func,
                name=name,
                description=description,
                handle_tool_error=True,
            )
            for func, name, description in specs
        ]
