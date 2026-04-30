# Personal AI Study Assistant

A local AI-powered study assistant that allows students to ask questions about their lecture notes, generate summaries, create quiz questions, and prepare for exams.

## Features

- Ask questions about lecture notes (RAG)
- Beginner-friendly explanations
- Topic summarization
- Quiz and flashcard generation
- Exam preparation support
- Local LLM using Ollama 
- Vector search using FAISS
- Agent-based routing with LangGraph
- FastAPI backend

## How It Works

1. Documents (TXT/PDF) are stored in `data/documents/`
2. Documents are split into chunks
3. Chunks are converted into embeddings
4. Embeddings are stored in a FAISS vector database
5. When a user asks a question:
   - Relevant chunks are retrieved
   - Context is passed to a local LLM (Ollama)
   - The model generates an answer based only on the provided context

---

## Tech Stack

- Python
- Ollama 
- LangChain
- LangGraph
- FAISS (Vector Database)
- Sentence Transformers
- FastAPI

## Setup and Run

### 1. Clone the repository
```bash 
git clone <your-repository-link>
cd personal-ai-study-assistant
```

### 2. Install the dependencies 
```bash
pip install -r requirements.txt
```

### 3. Install Ollama and pull model 
```
https://ollama.com/

```
#### run:
```bash
ollama pull llama3.1:8b
```

### 4. Add study material inside data/documents (.txt and .pdf files supported)

### 5. Build a vector database
```bash
python ingest.py
```

### 6. Run the API
```bash
uvicorn app:app --reload
```
### 7. Open in browser and test out queries based on the uploaded documents 


## Future improvement updates:

- Chat memory to be added 
- Frontend UI for better user experience
