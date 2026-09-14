from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
spliter=CharacterTextSplitter(
    separator="",
    chunk_size=10,
    chunk_overlap=1)

with open("document loader/test.txt", "r", encoding="utf-8") as f:
    text = f.read()

data = [Document(page_content=text)]
chunck=spliter.split_documents(data)
print(len(chunck ))
print(chunck[0].page_content )