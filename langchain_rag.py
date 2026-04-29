from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

#load document
loader = TextLoader("data/documents/ai.txt")
documents = loader.load()

#split document
splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)
docs = splitter.split_documents(documents)

#embeddings 
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

#vector store
db = FAISS.from_documents(docs, embeddings)

#retriever 
retriever = db.as_retriever(search_kwargs={"k": 2})

#load local llm 
llm = Ollama(model="llama3.1:8b")

#query 
question = "How are machine learning models trained?"

#retrieving relevant documents 
retrieved_docs = retriever.invoke(question)
context = "".join([doc.page_content for doc in retrieved_docs])

#Prompt 
prompt = f"""
Answer the user's question using only the context below. 

If the answer is not in the context, say: "I don't know the answer based on the provided context..."

Context: {context}
Question: {question}

Answer:
"""

answer = llm.invoke(prompt)

print("Quesrion:")
print(question)

print("\nRetrieved context:")
print(context)

print("\nAnswer:")
print(answer)

 



