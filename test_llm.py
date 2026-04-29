from langchain_community.llms import Ollama

llm = Ollama(model="llama3.1:8b")

prompt = "Explain RAG in 2 lines."
print(llm.invoke(prompt))