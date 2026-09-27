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
```

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

# 🧪 Tests

The pipeline takes its retriever and LLM as constructor arguments, so the
whole suite runs against fakes --- **no API keys, no network calls, no
cost**. 48 tests, 100% branch coverage of the `rag/` package, under a
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
└── test_api.py              ← Flask endpoints: status codes, upload, query
```

### Test doubles

`conftest.py` provides three fakes, all deterministic and offline:

  Double            Replaces                    Why
  ----------------- --------------------------- ---------------------------------
  `HashEmbeddings`  Hugging Face Inference API  Real embeddings need a token and aren't reproducible
  `FakeLLM`         Gemini                      Records the prompt it receives so tests can assert on the context
  `FakeRetriever`   Chroma retriever            Returns a fixed document set and records the question it was asked

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
suite on every push and pull request across Python 3.11, 3.12 and 3.13.
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
-   Page-number citations in answers
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
