from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_community.llms import Ollama

#defining graph state
class AgentState(TypedDict):
    question: str
    route: str
    result: str
    answer: str

llm = Ollama(model="llama3.1:8b")

def calculator(expression: str) -> str:
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Calculation error: {e}"
    
def doc_search(question: str) -> str:
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

#router node 
def router_node(state: AgentState) -> AgentState:
    question = state["question"]

    prompt = f"""
Decide which route should handle this query.

Routes:
calculator (for math problems)
doc_search (for user questions regarding AI, ML, neural networks, transformers, gradient descent)
general (for a conversation)

Return ONLY one word:
calculator
doc_search
general

Question: {question}
    """

    route = llm.invoke(prompt).strip().lower()

    if "calculator" in route:
        route = "calculator"
    elif "document" in route:
        route = "doc_search"
    else:
        route = "general"

    return {"route" : route}

#route decision function
def route_decision(state: AgentState) -> Literal["calculator", "document", "general"]:
    return state["route"]

#calculator node
def calculator_node(state: AgentState) -> AgentState:
    question = state["question"]

    expression = (question.lower().replace("what is", "").replace("calculate", "").replace("?", "").strip())

    result = calculator(expression)
    return {"result": result}

#document node 
def document_node(state: AgentState) -> AgentState:
    result = doc_search(state["question"])
    return {"result": result}

#general node 
def general_node(state: AgentState) -> AgentState:
    return {"result": "No tool required..."}

#final answer node
def final_node(state: AgentState) -> AgentState:
    prompt = f"""
Question: {state["question"]}
Result: {state["result"]}

Give an answer.
"""
    answer = llm.invoke(prompt)
    return {"answer": answer}

#building the graph
graph = StateGraph(AgentState)

graph.add_node("router", router_node)
graph.add_node("calculator", calculator_node)
graph.add_node("document", document_node)
graph.add_node("general", general_node)
graph.add_node("final_node", final_node)

graph.set_entry_point("router")

graph.add_conditional_edges(
    "router",
    route_decision,
    {
        "calculator": "calculator",
        "document": "document",
        "general" : "general",
    }
)

graph.add_edge("calculator", "final_node")
graph.add_edge("document", "final_node")
graph.add_edge("general", "final_node")
graph.add_edge("final_node", END)

app = graph.compile()

#test
questions = [
    "What is gradient descent?",
    "What is 12 * 8 + 4?",
    "Hello, what can you do?"
]

for q in questions:
    result = app.invoke({
        "question": q,
        "route": "",
        "result": "",
        "answer": ""
    })

    print("\nQuestion:", q)
    print("Route:", result["route"])
    print("Result:", result["result"])
    print("Answer:", result["answer"])











"""
Problems:
- The LLM is not correctly identifying the prompt as general, calculator or document searching

Fix:
- Update router_node (to be done later)
"""