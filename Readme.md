# 📚 DocuRAG --- PDF Question Answering with RAG

DocuRAG is a simple **Retrieval-Augmented Generation (RAG)** project
that allows users to ask questions about the contents of a PDF document.

The project is built around two main steps:

1.  **`create_database.py`** --- reads a PDF, splits it into chunks,
    creates embeddings, and stores them in ChromaDB.
2.  **`main.py`** --- loads the ChromaDB vector database, retrieves
    relevant document chunks, and uses an LLM to answer questions based
    on the retrieved context.

The project demonstrates the core workflow behind a document-based RAG
system.

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
├── create_database.py
├── main.py
├── requirements.txt
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

# ⚙️ `create_database.py`

The purpose of `create_database.py` is to convert the PDF document into
a searchable vector database.

### Step 1 --- Load the PDF

The project uses LangChain's `PyPDFLoader`:

``` python
data = PyPDFLoader("document loader/deeplearning.pdf")
docs = data.load()
```

This extracts the text and metadata from the PDF.

### Step 2 --- Split the Document

Large documents are divided into smaller chunks:

``` python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)
```

The overlap helps maintain context between neighboring chunks.

### Step 3 --- Generate Embeddings

The project uses the Hugging Face embedding model:

``` text
BAAI/bge-m3
```

Example:

``` python
emb_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3"
)
```

The embedding model converts each text chunk into a numerical vector.

### Step 4 --- Store Vectors in ChromaDB

The embeddings and document chunks are stored in ChromaDB:

``` python
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=emb_model,
    persist_directory="ChromaDB"
)
```

After running this script, the document can be searched semantically
instead of using simple keyword matching.

------------------------------------------------------------------------

# 🔎 `main.py`

The `main.py` file loads the previously created ChromaDB and provides an
interactive command-line RAG system.

### Step 1 --- Load the Embedding Model

The same embedding model used during database creation must be used when
querying the database:

``` python
emb_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3"
)
```

### Step 2 --- Load ChromaDB

The stored vector database is opened:

``` python
vectorstore = Chroma(
    persist_directory="ChromaDB",
    embedding_function=emb_model
)
```

### Step 3 --- Create the Retriever

The project uses **MMR (Maximal Marginal Relevance)** retrieval:

``` python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)
```

MMR attempts to retrieve information that is both:

-   Relevant to the question
-   Diverse enough to avoid returning nearly identical chunks

### Step 4 --- Send Context to the LLM

When the user asks a question, the system retrieves relevant chunks:

``` python
docs = retriever.invoke(query)
```

The retrieved text is then placed into the prompt:

``` text
Context:
<retrieved document chunks>

Question:
<user question>
```

The LLM is instructed to answer using the supplied context.

If the information is not available in the document, the prompt tells
the model to respond:

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

------------------------------------------------------------------------

# 🔐 Environment Variables

Create a `.env` file in the project root.

For the Google Gemini implementation, configure your Google API key:

``` env
GOOGLE_API_KEY=your_google_api_key
```

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
