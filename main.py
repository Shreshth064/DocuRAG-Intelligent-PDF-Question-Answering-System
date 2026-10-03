import argparse

from dotenv import load_dotenv

from rag import DocumentAgent, RAGPipeline
from rag.environment import require


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions about the indexed PDF.")
    parser.add_argument(
        "--agent",
        action="store_true",
        help="let the LLM choose and chain document-scoped tools "
        "(search, citations, summary, calculator) instead of a single "
        "retrieve-then-answer pass",
    )
    args = parser.parse_args()

    load_dotenv()
    require("GOOGLE_API_KEY", "HUGGINGFACEHUB_API_TOKEN")
    engine = DocumentAgent.from_config() if args.agent else RAGPipeline.from_config()

    print("rag system" + (" (agent mode)" if args.agent else ""))
    print("0 to exit")

    while True:
        query = input("you :")
        if query == "0":
            break

        answer = engine.ask(query)
        print(f"\n Ai:{answer.text}")
        if args.agent and answer.tools_used:
            print(f" (tools: {', '.join(answer.tools_used)})")


if __name__ == "__main__":
    main()
