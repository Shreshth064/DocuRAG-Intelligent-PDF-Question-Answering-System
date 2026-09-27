import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag import DocumentIngestor, KnowledgeBase, RAGConfig, RAGPipeline
from rag.environment import describe, missing

# Look for .env next to this script, regardless of the working directory
# `streamlit run` is launched from.
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)

# Uploaded PDFs get their own store so they never mix with the corpus
# built by create_database.py.
CONFIG = RAGConfig(persist_directory="chroma_db")

absent = missing("GOOGLE_API_KEY", "HUGGINGFACEHUB_API_TOKEN")
if absent:
    st.error(describe(absent))
    st.stop()


@st.cache_resource
def get_pipeline() -> RAGPipeline:
    return RAGPipeline.from_config(CONFIG)


st.set_page_config(page_title="RAG Book Assistant")

st.title("📚 RAG Book Assistant")
st.write("Upload a PDF and ask questions from the document")

uploaded_file = st.file_uploader("Upload a PDF book", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success("PDF uploaded successfully!")

    if st.button("Create Vector Database"):
        with st.spinner("Processing document..."):
            chunks = DocumentIngestor(CONFIG).ingest(file_path)
            KnowledgeBase(CONFIG).build(chunks)

        get_pipeline.clear()
        st.success(f"Vector database created from {len(chunks)} chunks!")

if KnowledgeBase(CONFIG).is_populated:
    st.divider()
    st.subheader("Ask Questions From the Book")

    query = st.text_input("Enter your question")

    if query:
        answer = get_pipeline().ask(query)

        st.write("### AI Answer")
        st.write(answer.text)

        if answer.pages:
            st.caption(f"Source pages: {', '.join(str(p) for p in answer.pages)}")
