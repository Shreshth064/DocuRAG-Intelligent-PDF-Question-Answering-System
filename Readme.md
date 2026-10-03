# 📚 DocuRAG --- PDF Question Answering with RAG

DocuRAG is a simple **Retrieval-Augmented Generation (RAG)** project
that allows users to ask questions about the contents of a PDF document.

The retrieval logic lives in a reusable **`rag/` package**, and three
thin entry points sit on top of it:

1.  **`create_database.py`** --- reads a PDF, splits it into chunks,
    creates embeddings, and stores them in ChromaDB.
2.  **`main.py`** --- a command-line chat over the stored document.
3.  **`app.py`** --- a Streamlit web app that does both: upload a PDF,
    build the store, then ask questions.

All three share the same engine, so the CLI and the web app can never
drift apart. See [Architecture](#️-architecture) for the class design.

------------------------------------------------------------------------

## 🚀 How It Works

``` text
                 PDF Document
                      │
                      ▼
              ┌───────────────┐
              │ PyPDFLoader   │
              └───────┬───────┘
                      │
                      ▼
          ┌──────────────────────┐
          │ Text Chunking        │
          │ chunk_size = 1000    │
          │ overlap = 200        │
          └──────────┬───────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │ BGE-M3 Embeddings    │
          │ Hugging Face         │
          └──────────┬───────────┘
                     │
                     ▼
             ┌──────────────┐
             │   ChromaDB   │
             │ Vector Store │
             └──────┬───────┘
                    │
             User Question
                    │
                    ▼
          ┌──────────────────────┐
          │ MMR Retriever        │
          │ k=4                  │
          │ fetch_k=10           │
          └──────────┬───────────┘
                     │
                     ▼
              Retrieved Context
                     │
                     ▼
          ┌──────────────────────┐
          │ LLM                  │
          │ Google Gemini        │
          └──────────┬───────────┘
                     │
                     ▼
                AI Answer
```

------------------------------------------------------------------------

## 🧠 What is RAG?

**Retrieval-Augmented Generation (RAG)** combines document retrieval
with a Large Language Model.

Instead of asking an LLM to answer a question only from its pretrained
knowledge, this project first searches a document for relevant
information.

The retrieved text is then provided to the LLM as context.

``` text
Question
   ↓
Search the vector database
   ↓
Retrieve relevant document chunks
   ↓
Add chunks to the prompt
   ↓
Send prompt to the LLM
   ↓
Generate answer
```

This makes the system useful for asking questions about books, notes,
research papers, manuals, and other PDF documents.

------------------------------------------------------------------------

# 📂 Project Structure

``` text
DocuRAG-Intelligent-PDF-Question-Answering-System/
│
├── rag/                     ← reusable engine (all the logic lives here)
│   ├── __init__.py
│   ├── config.py            ← RAGConfig
│   ├── ingestion.py         ← DocumentIngestor
│   ├── knowledge_base.py    ← KnowledgeBase
│   ├── pipeline.py          ← RAGPipeline, Answer
│   ├── tools.py             ← DocumentToolkit: document-scoped agent tools
│   ├── agent.py             ← DocumentAgent: tool-calling agent (opt-in)
│   └── api.py               ← Flask REST API (thin adapter over the engine)
│
├── create_database.py       ← entry point: build the vector store
├── main.py                  ← entry point: command-line chat
├── app.py                   ← entry point: Streamlit web app
│
├── tests/                   ← pytest suite (offline, no API keys)
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_ingestion.py
│   ├── test_knowledge_base.py
│   ├── test_pipeline.py
│   ├── test_tools.py
│   ├── test_agent.py
│   └── test_api.py
│
├── pyproject.toml           ← pytest + coverage configuration
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
│
├── document loader/
│   └── PDF documents
│
└── ChromaDB/
    └── Generated vector database
```

> Local virtual environments, `.env` files, and generated databases
> should not be uploaded to GitHub.

------------------------------------------------------------------------

# 🏗️ Architecture

All of the logic lives in the `rag/` package. The three scripts at the
root (`create_database.py`, `main.py`, `app.py`) are thin frontends that
share one engine, so the CLI and the web app can never drift apart.

``` text
        create_database.py     main.py        app.py
          (build store)        (CLI)       (Streamlit)
                 │               │              │
                 └───────────────┼──────────────┘
                                 ▼
                        ┌─────────────────┐
                        │   RAGPipeline   │  ask() -> Answer
                        └────────┬────────┘
                       composes  │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
        ┌─────────────────┐            ┌──────────────────┐
        │  KnowledgeBase  │            │  ChatGoogle-     │
        │  embeddings +   │            │  GenerativeAI    │
        │  Chroma store   │            └──────────────────┘
        └────────┬────────┘
                 ▲
        ┌────────┴────────┐          ┌──────────────────┐
        │ DocumentIngestor│          │    RAGConfig     │
        │  PDF -> chunks  │◄─────────│  every tunable   │
        └─────────────────┘          └──────────────────┘
```

  Class               Responsibility
  ------------------- ----------------------------------------------------
  `RAGConfig`         Frozen dataclass holding every tunable (models, paths, chunk size, retrieval settings)
  `DocumentIngestor`  Loads a PDF and splits it into overlapping chunks
  `KnowledgeBase`     Owns the embedding model and the Chroma vector store; builds it, opens it, exposes a retriever
  `RAGPipeline`       Composes retriever + prompt + LLM into `ask()`
  `Answer`            Immutable result carrying the generated text and the source documents

### Design notes

-   **Encapsulation** --- callers never touch `Chroma` or the embedding
    client directly; `KnowledgeBase` opens the store lazily on first use.
-   **Composition over inheritance** --- `RAGPipeline` *has a* retriever
    and *has an* LLM rather than inheriting from either.
-   **Dependency injection** --- `RAGPipeline(retriever=..., llm=...)`
    takes its collaborators as arguments. `from_config()` is a factory
    classmethod that wires the real ones, while tests pass fakes. This is
    what makes the pipeline testable without an API key.
-   **One source of truth** --- every magic number lives in `RAGConfig`.
    The frontends override only what they need
    (`RAGConfig(persist_directory="chroma_db")` in the Streamlit app).

------------------------------------------------------------------------

# 🤖 Agentic Tools

Besides the fixed retrieve-then-answer `RAGPipeline`, DocuRAG has an
**opt-in agentic path**. A LangChain tool-calling agent
(`create_tool_calling_agent` + `AgentExecutor`, over the same Gemini model
named in `RAGConfig.llm_model`) decides for itself which tools to call and
in what order. For example, it might search the document, pull the
citations, and then run the calculator on the figures it found, all before
it answers.

  Tool                        What it does
  --------------------------- ------------------------------------------------------------
  `search_documents(query)`   Retrieves the most relevant chunks from the vector store and returns their text
  `get_citations(query)`      Returns the provenance of those chunks: page numbers, source file and chunk ids
  `summarize_document(topic)` Retrieves more broadly (`RAGConfig.summary_k` chunks) and has the LLM summarise **only those chunks**. With no matching chunks it abstains without calling the LLM
  `calculator(expression)`    Evaluates arithmetic on figures from the document with a **safe AST evaluator**: numbers, `+ - * / **`, parentheses and unary minus only. There is no `eval()`. Names, calls, attribute access and oversized powers are rejected, and the error is handed back to the agent

Use it from the CLI with `python main.py --agent`, or over HTTP with
`POST /agent` (see the REST API endpoints below). `/query` and the
Streamlit app are unchanged.

``` python
from rag import DocumentAgent

answer = DocumentAgent.from_config().ask("By how much did revenue grow?")
answer.text        # the grounded answer
answer.tools_used  # e.g. ["search_documents", "get_citations", "calculator"]
```

### Design decision: tools are document-scoped

RAG exists so that an answer can be traced back to the document. An agent
with open-ended tools would break that guarantee without anyone noticing.
Once a model can mix retrieved chunks with web results, a reader can no
longer tell which sentence came from the PDF and which came from the
internet. Every tool here is therefore **scoped to the loaded
document(s)**:

-   The retrieval tools only read the Chroma store. `summarize_document`
    shows its LLM nothing but the retrieved chunks. `calculator` only
    does arithmetic and has no access to data.
-   The agent's system prompt tells it to answer **only** from tool
    output, never from background knowledge. If the tools don't surface
    the answer, it must reply with the same abstention message the
    pipeline uses (*"I could not find the answer in the document."*).
-   **Web search was deliberately excluded.** If it is ever added, it
    should be a clearly labelled, abstain-first fallback: the agent would
    first say that the document doesn't contain the answer, and any
    outside result would be marked as not coming from the document. It
    would never be blended into a grounded answer.

### Design notes

-   `DocumentToolkit(retriever, llm, summary_retriever=None)` takes its
    collaborators by injection, like `RAGPipeline`.
    `from_knowledge_base()` wires the real ones, and `as_tools()` exposes
    them as LangChain `StructuredTool`s.
-   `build_agent(llm, tools)` returns the `AgentExecutor`. The loop is
    capped at `max_iterations` so a confused agent can't spin forever.
    `DocumentAgent.from_config()` / `from_knowledge_base()` are factories
    consistent with the rest of the package.
-   `DocumentAgent.ask()` returns an immutable `AgentAnswer` that carries
    the text plus every `ToolCall` (tool, input, output), so the
    reasoning trace can be audited.
-   `AgentExecutor` and `create_tool_calling_agent` moved to the
    `langchain-classic` package in LangChain 1.x. It is the only new
    dependency, and it was already being installed through
    `langchain-community`.

------------------------------------------------------------------------

# ⚙️ `create_database.py`

The purpose of `create_database.py` is to convert the PDF document into
a searchable vector database.

### Step 1 --- Load and Split the PDF

`DocumentIngestor` wraps LangChain's `PyPDFLoader` and
`RecursiveCharacterTextSplitter` behind one call:

``` python
chunks = DocumentIngestor(config).ingest("document loader/deeplearning.pdf")
```

Internally that is still load-then-split, and both steps stay available
separately (`.load()` and `.split()`) when you need them. The chunk size
and overlap come from `RAGConfig`, so the overlap that preserves context
between neighbouring chunks is configured in exactly one place.

### Step 2 --- Embed and Store the Chunks

`KnowledgeBase` owns both the embedding model and the Chroma store, so a
single call embeds the chunks and persists them:

``` python
KnowledgeBase(config).build(chunks)
```

Embeddings are generated through the **Hugging Face Inference API** using
`BAAI/bge-m3`, so the model is never downloaded to your machine --- it
only needs a `HUGGINGFACEHUB_API_TOKEN`:

``` python
HuggingFaceEndpointEmbeddings(model="BAAI/bge-m3")
```

The embedding model converts each text chunk into a numerical vector, and
Chroma writes those vectors to `config.persist_directory`.

After running this script, the document can be searched semantically
instead of using simple keyword matching.

------------------------------------------------------------------------

# 🔎 `main.py`

The `main.py` file loads the previously created ChromaDB and provides an
interactive command-line RAG system.

### Step 1 --- Build the Pipeline

Because `RAGConfig` holds the embedding model, the store location and the
retrieval settings, the whole system is wired up in one line:

``` python
pipeline = RAGPipeline.from_config()
```

This guarantees the query side uses the *same* embedding model as the
build side --- a mismatch there silently returns nonsense, and keeping
both in one config makes it impossible.

### Step 2 --- Retrieval

Under the hood `KnowledgeBase.as_retriever()` uses
**MMR (Maximal Marginal Relevance)** with the settings from the config
(`k=4`, `fetch_k=10`, `lambda_mult=0.5`).

MMR attempts to retrieve information that is both:

-   Relevant to the question
-   Diverse enough to avoid returning nearly identical chunks

### Step 3 --- Ask a Question

``` python
answer = pipeline.ask("What is deep learning?")

print(answer.text)     # the generated answer
print(answer.pages)    # page numbers the answer was drawn from
```

`ask()` retrieves the chunks, formats them into the prompt, calls Gemini,
and returns an `Answer` object carrying both the text and the source
documents it used --- so the caller can cite pages without re-running the
retrieval.

The prompt instructs the LLM to answer using only the supplied context.
If the information is not in the document, the model is told to respond:

``` text
I could not find the answer in the document.
```

------------------------------------------------------------------------

# 🛠️ Tech Stack

  Technology                       Purpose
  -------------------------------- ---------------------------------
  Python                           Programming language
  LangChain                        RAG pipeline
  PyPDFLoader                      PDF text extraction
  RecursiveCharacterTextSplitter   Document chunking
  Hugging Face                     Embedding model integration
  BAAI/bge-m3                      Text embeddings
  ChromaDB                         Vector database
  Google Gemini                    LLM for answer generation
  python-dotenv                    Environment variable management
  Streamlit                        Web interface
  Flask                            REST API
  pytest                           Test suite
  pytest-cov                       Coverage measurement
  GitHub Actions                   Continuous integration

------------------------------------------------------------------------

# 📦 Installation

### 1. Clone the repository

``` bash
git clone https://github.com/Shreshth064/DocuRAG-Intelligent-PDF-Question-Answering-System.git
```

``` bash
cd DocuRAG-Intelligent-PDF-Question-Answering-System
```

### 2. Create a virtual environment

Windows:

``` bash
python -m venv .venv
```

Activate it:

``` bash
.venv\Scripts\activate
```

macOS/Linux:

``` bash
python3 -m venv .venv
```

``` bash
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

To also install the test tooling:

``` bash
pip install -r requirements-dev.txt
```

------------------------------------------------------------------------

# 🔐 Environment Variables

Create a `.env` file in the project root:

``` env
GOOGLE_API_KEY=your_google_api_key
HUGGINGFACEHUB_API_TOKEN=your_hugging_face_token
```

-   `GOOGLE_API_KEY` --- used by Gemini to generate answers.
-   `HUGGINGFACEHUB_API_TOKEN` --- used to call the Hugging Face
    Inference API for embeddings. Because embeddings run through the API,
    the `BAAI/bge-m3` model is **not** downloaded to your machine.
    Get a token at <https://huggingface.co/settings/tokens>.

**Never upload `.env` to GitHub.**

------------------------------------------------------------------------

# ▶️ Run the Project

## Step 1 --- Create the Vector Database

First run:

``` bash
python create_database.py
```

This will:

``` text
PDF
 ↓
Extract text
 ↓
Split into chunks
 ↓
Generate BGE-M3 embeddings
 ↓
Store embeddings in ChromaDB
```

After this step, the vector database will be available locally.

## Step 2 --- Start the Question Answering System

Run:

``` bash
python main.py
```

You should see:

``` text
rag system
0 to exit
you :
```

Ask a question about the PDF:

``` text
you : What is deep learning?
```

The system retrieves relevant sections from the document and sends them
to the LLM.

To exit:

``` text
0
```

To let the LLM choose and chain the document-scoped tools instead (see
[Agentic Tools](#-agentic-tools)):

``` bash
python main.py --agent
```

## Step 3 --- (Optional) Run the Web App

``` bash
streamlit run app.py
```

Upload a PDF, click **Create Vector Database**, then ask questions. The
web app uses its own store (`chroma_db/`) so uploads never mix with the
corpus built by `create_database.py`.

## Step 4 --- (Optional) Run the REST API

``` bash
python -m rag.api
```

This starts a Flask server on port 8000. It is a thin adapter over the
same `rag/` engine --- no pipeline logic is duplicated. The embedding
client, the Chroma store and the LLM are constructed **once at startup**,
never per request; after an upload, only the lightweight pipeline wrapper
is re-pointed at the new store.

> The server constructs the Gemini client eagerly at startup, so it
> **fails fast** if `GOOGLE_API_KEY` is missing rather than serving while
> unable to answer.

### Endpoints

  Method &amp; path    Request                        Success        Errors
  ---------------- ------------------------------ -------------- -------------------------
  `GET /health`    none                           `200` `{"status":"ok"}`  ---
  `POST /upload`   `multipart/form-data`, `file`  `201` `{"chunks": N}`    `400` no file · `415` not a PDF · `413` too large · `500` ingest failed
  `POST /query`    JSON `{"question": "..."}`      `200` answer + `pages`  `400` empty question · `404` no store yet · `500` query failed
  `POST /agent`    JSON `{"question": "..."}`      `200` answer + `tools_used` + `steps`  `400` empty question · `404` no store yet · `500` agent failed

`/health` touches no dependencies, so it is safe for container
healthchecks. Every response --- including errors --- is JSON; a single
error handler converts the API's typed exceptions into a structured
`{"error": "..."}` body and never leaks a stack trace.

### Example

``` bash
# 1. build the store from a PDF
curl -F "file=@book.pdf" http://localhost:8000/upload
# -> {"chunks": 42}

# 2. ask a question
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"question": "What is deep learning?"}'
# -> {"answer": "...", "pages": [3, 7], "num_sources": 4}

# 3. or let the agent choose and chain document-scoped tools
curl -X POST http://localhost:8000/agent \
     -H "Content-Type: application/json" \
     -d '{"question": "By how much did revenue grow?"}'
# -> {"answer": "...", "tools_used": ["search_documents", "calculator"],
#     "steps": [{"tool": "search_documents", "input": {"query": "revenue"}}, ...]}
```

`/agent` is built per request, which is cheap because it only binds the
tools, so it always wraps the current store. It is not cached.

### Safety notes

-   Uploaded files are validated by their **magic bytes** (`%PDF-`), not
    by the client-supplied name or content type.
-   Filenames are sanitised with `werkzeug.utils.secure_filename`, and
    the temp file is written to an OS temp path and deleted in a
    `finally` block whether ingestion succeeds or fails.
-   `MAX_CONTENT_LENGTH` caps the request body so oversized uploads are
    rejected (`413`) before being read into memory.

### Application-factory pattern

`create_app(config=None, knowledge_base=None, llm=None)` mirrors the
project's dependency-injection style: production passes nothing and the
real components are built; tests inject a `KnowledgeBase` over fake
embeddings and a fake LLM, so the entire API is exercised offline in
`tests/test_api.py`.

------------------------------------------------------------------------

# 🐳 Running with Docker

Both frontends ship from a single image. Because embeddings now go
through the Hugging Face Inference API, no model is baked in or
downloaded at runtime --- the only state is the Chroma persist directory.
Compose also starts a **Redis** service that caches answered queries; the
`api` and `ui` services depend on it, but it is optional at runtime (see
[Query cache](#query-cache-redis) below).

### Prerequisites

Install Docker. `docker-buildx` is not strictly required --- the
Dockerfile avoids BuildKit-only features so it builds with the classic
builder too --- but it makes rebuilds noticeably faster:

``` bash
sudo apt update
sudo apt install -y docker.io docker-compose-v2 docker-buildx
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"   # takes effect after you log out and back in
```

Until you re-login, prefix commands with `sudo`.

Create `.env` in the project root first; compose injects it at runtime:

``` env
GOOGLE_API_KEY=your_google_api_key
HUGGINGFACEHUB_API_TOKEN=your_hugging_face_token
```

`.env` is listed in `.dockerignore`, and no `ARG`/`ENV` carries a key, so
**nothing secret is ever baked into the image**. The keys arrive only at
runtime, via `env_file`. Verify against a fresh container (no `env_file`):

``` bash
docker run --rm docurag-api:latest printenv | grep -iE 'api_key|token'
# (no output)
```

### Build and run

``` bash
docker compose build
docker compose up          # API on :8000, Streamlit on :8501, Redis on :6379

curl http://localhost:8000/health
# {"status":"ok"}
```

`api` and `ui` declare `depends_on: redis` with `condition: service_healthy`,
so compose waits for Redis to pass its `redis-cli ping` health check before
starting them.

Each app service builds to its own image tag (`docurag-api`, `docurag-ui`,
`docurag-test`) so a parallel build never races on a shared tag; `redis`
runs the stock `redis:7-alpine` image.

| Service | Port | Image | Command |
| --- | --- | --- | --- |
| `redis` | 6379 | `redis:7-alpine` | `redis-server --appendonly yes` |
| `api` | 8000 | `docurag-api` | `python -m rag.api` (the image default) |
| `ui` | 8501 | `docurag-ui` | `streamlit run app.py` |
| `test` | --- | `docurag-test` | `pytest --cov` (profile `test`) |

### Query cache (Redis)

The `/query` endpoint caches each answer in Redis, keyed on a hash of the
question **and** a fingerprint of the active vector store. A repeated
question against an unchanged corpus is served straight from the cache,
skipping retrieval and the LLM call entirely; re-uploading a document
changes the fingerprint, so stale answers are never returned.

The connection is configured with `REDIS_URL` --- `redis://redis:6379` in
compose, defaulting to `redis://localhost:6379` elsewhere. The cache is
**best-effort**: if Redis is unreachable the API logs a warning and answers
uncached. A cache outage never fails a request or crashes startup, so you
can also run the API with no Redis at all. The cache survives a restart
because Redis persists to the `redis-data` volume (`--appendonly yes`).

### Run the tests in the image

The `test` stage layers the dev dependencies and the suite on top of the
runtime image, so CI runs exactly what ships:

``` bash
docker compose run --rm test
```

It is behind a compose profile, so a plain `docker compose up` will not
start it. No `env_file` is attached on purpose --- the suite must stay
runnable with no credentials.

### The persist volumes

Chroma data lives in **named volumes**, so it survives
`docker compose down` and container restarts. There are two, because the
two frontends use different persist directories:

| Volume | Mounted at | Used by |
| --- | --- | --- |
| `api-store` | `/app/ChromaDB` | `api` (the `RAGConfig` default) |
| `ui-store` | `/app/chroma_db` | `ui` (`app.py` overrides the path) |

Keeping them separate preserves current behaviour --- Streamlit uploads
stay out of the corpus built by `create_database.py` --- and avoids two
processes writing one SQLite file. To make both services share one store,
point them at the same named volume; expect possible `database is locked`
errors under concurrent writes.

Wipe the stored vectors with:

``` bash
docker compose down -v
```

### Image design notes

-   **Base:** `python:3.12-slim-bookworm`, pinned. 3.12 sits in the
    middle of the 3.11--3.14 CI matrix and has the most dependable
    prebuilt wheels for `chromadb`'s tree (`onnxruntime`,
    `pydantic-core`); on 3.13/3.14 a missing wheel forces a source build.
-   **Multi-stage:** a builder installs dependencies into `/opt/venv`
    with `uv` (pinned to 0.12.19); the runtime stage copies only that
    venv, leaving `uv`, `build-essential` and every download cache
    behind.
-   **Layer caching:** `requirements.txt` is copied and installed
    *before* the source, so editing code does not rebuild dependencies.
-   **Non-root:** runs as `appuser` (uid 1000). The persist directories
    are created and chowned in the image so named volumes inherit that
    ownership rather than defaulting to root.
-   **Healthcheck:** `curl -fsS /health`. `curl` is the only apt package
    in the runtime stage, since the healthcheck needs it at run time.
-   **Expected size:** roughly **0.9 GB**. That is the dependency tree,
    not a packaging leak --- Streamlit pulls `pyarrow` (152 MB),
    `pandas` and `pydeck`, while Chroma pulls `onnxruntime` (62 MB),
    `chromadb_rust_bindings` and `kubernetes`. Together that is ~760 MB
    of site-packages before the base image. There is no `torch`. If the
    API image needs to be smaller, the lever is splitting Streamlit into
    its own stage so the API image drops ~215 MB --- not `.dockerignore`,
    which is already excluding the venv, `.git`, vector stores and PDFs.

------------------------------------------------------------------------

# 🧪 Tests

The pipeline takes its retriever and LLM as constructor arguments, so the
whole suite runs against fakes --- **no API keys, no network calls, no
cost**. 146 tests, 100% branch coverage of the `rag/` package, under a
second to run.

### Install and run

``` bash
pip install -r requirements-dev.txt
pytest
```

With a coverage report:

``` bash
pytest --cov
```

### Layout

``` text
tests/
├── conftest.py              ← shared fixtures and offline test doubles
├── test_config.py           ← RAGConfig: defaults, immutability, overrides
├── test_ingestion.py        ← DocumentIngestor: chunking, overlap, real PDF
├── test_knowledge_base.py   ← KnowledgeBase: build, persist, reopen, retrieve
├── test_pipeline.py         ← RAGPipeline + Answer: prompt, sources, pages
├── test_tools.py            ← DocumentToolkit: safe calculator, grounded tools
├── test_agent.py            ← DocumentAgent: scripted tool calls end to end
└── test_api.py              ← Flask endpoints: status codes, upload, query
```

### Test doubles

`conftest.py` provides these fakes, all deterministic and offline:

  Double            Replaces                    Why
  ----------------- --------------------------- ---------------------------------
  `HashEmbeddings`  Hugging Face Inference API  Real embeddings need a token and aren't reproducible
  `FakeLLM`         Gemini                      Records the prompt it receives so tests can assert on the context
  `FakeRetriever`   Chroma retriever            Returns a fixed document set and records the question it was asked
  `ScriptedChatModel` Gemini (tool calling)     Replays scripted tool calls and answers so a real `AgentExecutor` runs offline

### Markers

Tests that write a real Chroma store to a temp directory are marked
`integration` (still offline --- they use `HashEmbeddings`):

``` bash
pytest -m "not integration"   # fast unit tests only
pytest -m integration         # storage round-trip tests only
```

### Strictness

`pyproject.toml` sets `filterwarnings = error`, so a new deprecation from
LangChain or Chroma **fails the build** instead of scrolling past. Known
third-party warnings are explicitly allow-listed with a comment
explaining each one.

### Continuous integration

[`.github/workflows/tests.yml`](.github/workflows/tests.yml) runs the
suite on every push and pull request across Python 3.11, 3.12, 3.13 and 3.14.
No secrets are configured in CI on purpose --- if a test ever needs an
API key, the build breaks, which keeps the suite honest.

------------------------------------------------------------------------

# 🔄 Complete Workflow

``` text
             create_database.py
                    │
                    ▼
              Load PDF file
                    │
                    ▼
              Split into chunks
                    │
                    ▼
             Create embeddings
                    │
                    ▼
                ChromaDB
                    │
                    │
                    ▼
                 main.py
                    │
                    ▼
              User asks question
                    │
                    ▼
             MMR retrieves chunks
                    │
                    ▼
              Build RAG prompt
                    │
                    ▼
             Google Gemini LLM
                    │
                    ▼
                AI Answer
```

------------------------------------------------------------------------

# 🎯 Key Concepts Demonstrated

This project demonstrates several important concepts in modern AI
application development:

**AI / retrieval**

-   Retrieval-Augmented Generation (RAG)
-   Vector embeddings
-   Semantic search
-   Vector databases
-   Document chunking
-   Similarity retrieval
-   Maximal Marginal Relevance (MMR)
-   Prompt engineering
-   LLM integration
-   PDF document processing

**Software design**

-   Object-oriented design (encapsulation, composition, single
    responsibility)
-   Dependency injection and factory methods
-   Immutable value objects (`RAGConfig`, `Answer` as frozen dataclasses)
-   Separation of concerns --- one engine, three interchangeable frontends
-   Centralised configuration instead of scattered constants
-   Unit testing with pytest --- fixtures, markers, parametrised doubles
-   100% branch coverage of the core package, enforced offline in CI
-   Warnings-as-errors so upstream deprecations fail the build
-   Type hints throughout

------------------------------------------------------------------------

# ⚠️ Limitations

-   Currently designed around PDF documents.
-   The vector database is stored locally.
-   The database needs to be regenerated when the source document
    changes.
-   Scanned/image-only PDFs may require OCR.
-   The quality of answers depends on PDF extraction, chunking,
    retrieval, and the LLM.
-   Gemini API usage may have usage limits or costs depending on the
    provider's current plan.

------------------------------------------------------------------------

# 🔮 Future Improvements

Possible improvements include:

-   Support for multiple PDFs
-   Automatic document indexing
-   Chat history
-   Streaming responses
-   OCR for scanned PDFs
-   Better document management
-   Web-based interface
-   Cloud-hosted vector databases
-   Retrieval evaluation and RAG benchmarking
-   Support for multiple LLM providers

------------------------------------------------------------------------

# 👨‍💻 Author

**Shreshth**

GitHub: `https://github.com/Shreshth064`

------------------------------------------------------------------------

⭐ If you find this project useful, consider starring the repository.
