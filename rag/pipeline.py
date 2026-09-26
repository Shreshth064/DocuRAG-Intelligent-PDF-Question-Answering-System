from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_google_genai import ChatGoogleGenerativeAI

from .config import RAGConfig
from .knowledge_base import KnowledgeBase

NOT_FOUND_MESSAGE = "I could not find the answer in the document."

DEFAULT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI assistant.\n\n"
            "Use ONLY the provided context to answer the question.\n\n"
            "If the answer is not present in the context, "
            f'say: "{NOT_FOUND_MESSAGE}"',
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion:\n{question}",
        ),
    ]
)


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[Document] = field(default_factory=list)

    @property
    def pages(self) -> list[int]:
        return sorted(
            {
                doc.metadata["page"]
                for doc in self.sources
                if "page" in doc.metadata
            }
        )


class RAGPipeline:
    """Composes a retriever, a prompt and an LLM into a question-answering step."""

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: BaseLanguageModel,
        prompt: ChatPromptTemplate = DEFAULT_PROMPT,
    ):
        self._retriever = retriever
        self._llm = llm
        self._prompt = prompt

    @classmethod
    def from_config(cls, config: RAGConfig | None = None) -> "RAGPipeline":
        config = config or RAGConfig()
        return cls(
            retriever=KnowledgeBase(config).as_retriever(),
            llm=ChatGoogleGenerativeAI(model=config.llm_model),
        )

    def retrieve(self, question: str) -> list[Document]:
        return self._retriever.invoke(question)

    def ask(self, question: str) -> Answer:
        sources = self.retrieve(question)
        context = "\n\n".join(doc.page_content for doc in sources)
        response = self._llm.invoke(
            self._prompt.invoke({"context": context, "question": question})
        )
        return Answer(text=response.content, sources=sources)
