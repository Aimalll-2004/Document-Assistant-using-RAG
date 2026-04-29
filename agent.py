"""
Creating a separate calculator (for simple math) and document search tool for the agent to choose from
"""

from langchain_community.llms import Ollama
from langchain_core.tools import tool 

llm = Ollama(model="llama3.1:8b")

#calculator 
@tool 
def calculator(expression: str) -> str:
    """Calculate a maths expression."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error calculating: {e}"
    
@tool 
def doc_search(question: str) -> str:
    """Search a knowledge base about AI and machine learning"""
    #simple knowledge base for now 
    knowledge = {
        "machine learning": "Machine learning allows computers to learn patterns from data.",
        "gradient descent": "Gradient descent is an optimization algorithm used to train machine learning models.",
        "transformers": "Transformers are neural network architectures used in NLP and LLMs.",
        "neural networks": "Neural networks are models inspired by the human brain."
    }

    question = question.lower()

    for key, value in knowledge.items():
        if key in question:
            return value 

    return "I could not find relevant information..."    

#agent router
def agent(question: str) -> str:
    prompt = f"""
Decide which tool should be used based on user's question.

Available tools:
1. Calculator (for math problems)
2. doc_search (for user questions regarding AI, ML, neural networks, transformers)

Return ONLY one word:
calculator
doc_search

Question:{question}
"""
    
    tool = llm.invoke(prompt).strip().lower()
    print("Tool chosen:", tool)

    if "calculator" in tool:
        expression = question.replace("calculate", "").replace("what is", "").strip()
        result = calculator.invoke(expression)

    elif "doc_search" in tool:
        result = doc_search.invoke(question)

    final_prompt = f"""
User Question: {question}

Result: {result}

Give an answer please.
"""
    return llm.invoke(final_prompt)

questions = [
    "What is gradient descent?",
    "What is 50 / 5?",
]

for q in questions:
    print("\nQuestion:", q)
    print("Answer:", agent(q))

    