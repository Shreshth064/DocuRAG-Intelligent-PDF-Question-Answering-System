from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
loader = PyPDFLoader("document loader/gru.pdf")

documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
slitted_documents = splitter.split_documents(documents)

print(len(slitted_documents))

