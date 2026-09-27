from dotenv import load_dotenv

from rag import RAGPipeline
from rag.environment import require


def main() -> None:
    load_dotenv()
    require("GOOGLE_API_KEY", "HUGGINGFACEHUB_API_TOKEN")
    pipeline = RAGPipeline.from_config()

    print("rag system")
    print("0 to exit")

    while True:
        query = input("you :")
        if query == "0":
            break

        answer = pipeline.ask(query)
        print(f"\n Ai:{answer.text}")


if __name__ == "__main__":
    main()
