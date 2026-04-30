from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "vector_db",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(search_kwargs={"k": 3})

llm = Ollama(model="llama3.1:8b")


def rag_answer(question: str) -> str:
    retrieved_docs = retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""
You are a AI study Assistant for University students.

Use ONLY the provided context to answer the questions:
If the context does not contain enough information, say:
"I don't know based on the provided lecture notes."


Your job:
- explain concepts clearly and simply
- give step-by-step explanations when useful
- connect ideas to examples
- mention if the context does not contain enough information
- do NOT invent facts outside the context

If the student asks for:
- a summary, provide a concise summary
- quiz questions, create questions only from the context
- flashcards, create Q/A flashcards only from the context
- exam prep, highlight important concepts from the context

Context:
{context}

Question:
{question}

Answer:
"""

    return llm.invoke(prompt)