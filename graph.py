from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_community.llms import Ollama
from rag_tool import rag_answer

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

#router node 
def router_node(state: AgentState) -> AgentState:
    question = state["question"].lower()

    math_symbols = ["+", "-", "*", "/", "calculate"]

    greetings = ["hello", "hi", "hey", "what can you do"]

    if any(symbol in question for symbol in math_symbols):
        route = "calculator"
    elif any(greeting in question for greeting in greetings):
        route = "general"
    else:
        route = "document"

    return {"route": route}

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
    result = rag_answer(state["question"])
    return {"result": result}

#general node 
def general_node(state: AgentState) -> AgentState:
    return {"result": "No tool required..."}

#final answer node
def final_node(state: AgentState) -> AgentState:
    if state["route"] == "document":
        return {"answer": state["result"]}

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
