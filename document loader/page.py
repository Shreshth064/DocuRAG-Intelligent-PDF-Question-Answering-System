from langchain_community.document_loaders import WebBaseLoader

url="https://www.apple.com/in/"
web_loader = WebBaseLoader(url)

web_data = web_loader.load()
print(web_data[0].page_content)

