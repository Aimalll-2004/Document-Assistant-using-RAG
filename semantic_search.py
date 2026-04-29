from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

documents = [
    "Machine learning allows computers to learn patterns from data.",
    "Neural networks are models inspired by the human brain.",
    "Pizza dough is made from flour, water, yeast, and salt.",
    "Gradient descent is an optimization algorithm used to train models.",
    "Transformers are neural network architectures used in NLP and LLMs."
]

model = SentenceTransformer("all-MiniLM-L6-v2") #load embedding model 

doc_embeddings = model.encode(documents) #documents to vectors 

#create faiss index 
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

#adding document vectors to FAISS 
index.add(np.array(doc_embeddings).astype("float32"))

#Query 
query = "How do AI Models learn from data?"
query_embedding = model.encode([query])

#Search for the most similar documents 
distances, indices= index.search(np.array(query_embedding).astype("float32"), k=2)

print("Query:", query)
print("\nMost relevant documents:")

for i in indices[0]:
    print("-", documents[i])