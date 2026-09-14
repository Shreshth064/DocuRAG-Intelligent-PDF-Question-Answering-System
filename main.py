from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()
emb_model=HuggingFaceEmbeddings(
     model_name="BAAI/bge-m3"
)
vectorstore=Chroma(
    persist_directory="ChromaDb",
    embedding_function=emb_model
)
retriever=vectorstore.as_retriever(
    search_type='mmr',
    search_kwargs={
        "k":4,
        "fetch_k":10,
        "lambda_mult":0.5 
    }
    
)
llm=ChatGoogleGenerativeAI(model="gemini-3.6-flash")
  
prompt=ChatPromptTemplate.from_messages([
    ("system",""" You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
     """),("human","""context:{context}
           question:{question}""")
])

print("rag system")
print ("0 to exit")
while True:
    query=input("you :")
    if query =="0":
        break
    docs=retriever.invoke(query)
    context="".join([
        doc.page_content for doc in docs
    ])
    
    fianl_prompt=prompt.invoke({"context":context,
                               "question":query})
    reponse=llm.invoke(fianl_prompt)
    
    print(f"\n Ai:{reponse.text}" )
  