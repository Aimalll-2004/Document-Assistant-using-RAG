"""Difference from what was done on day 2 in semantic_search.py?

BEFORE: question -> relevant document retrieved 
AFTER: question -> relevant document retrieved -> sent to LLM -> answer 
"""
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama
import faiss 
import numpy as np

documents = [
      "Machine learning allows computers to learn patterns from data.",
    "Neural networks are models inspired by the human brain and are used for pattern recognition.",
    "Pizza dough is made from flour, water, yeast, and salt.",
    "Gradient descent is an optimization algorithm used to train machine learning models.",
    "Transformers are neural network architectures used in NLP and large language models."
]

#create embeddings 
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embedding_model.encode(documents)

#store vectors in FAISS 
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings).astype("float32"))

#load local LLM
llm = Ollama(model="llama3.1:8b")

#query 
question = "What is quantum computing?"

#retrieve relevant documents 
query_embedding = embedding_model.encode([question])

distances, indices = index.search(
    np.array(query_embedding).astype("float32"), k=1
)

retrieved_docs = [documents[i] for i in indices[0]]

context = "\n".join(retrieved_docs)

#build prompt 
prompt = f""" 
Answer the user's question using only the context below. 

If the answer is not in the context, say: "I don't know the answer based on the provided context..."

Context: {context}
Question: {question}

Answer:
"""

answer = llm.invoke(prompt)

print("Question:")
print(question)

print("\nRetreived Context:")
print(context)

print("\nAnswer:")
print(answer)


