from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
load_dotenv()

data=PyPDFLoader("document loader/deeplearning.pdf")
da = data.load()
splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunk=splitter.split_documents(documents=da)



emb_model=HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
vectorstore=Chroma.from_documents(documents=chunk,embedding=emb_model,persist_directory="ChromaDB")