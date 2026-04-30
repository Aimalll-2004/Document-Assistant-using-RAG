from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

all_docs = []


#TXT loader
txt_loader = DirectoryLoader(
    "data/documents",
    glob="**/*.txt",
    loader_cls=TextLoader
)
all_docs.extend(txt_loader.load())

#PDF loader
pdf_loader = DirectoryLoader(
    "data/documents",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)
all_docs.extend(pdf_loader.load())

splitter = CharacterTextSplitter(
    chunk_size=750,
    chunk_overlap=100
)

docs = splitter.split_documents(all_docs)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = FAISS.from_documents(docs, embeddings)

db.save_local("vector_db")

print("Vector database saved successfully.")