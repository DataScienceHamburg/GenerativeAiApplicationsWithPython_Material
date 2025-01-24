#%% packages
from pprint import pprint
from typing_extensions import TypedDict
import random
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from IPython.display import Image, display
from rich.console import Console
from rich.markdown import Markdown
console = Console()

#%% LLM
llm = ChatGroq(model="gemma2-9b-it")

# State with graph_state
class State(TypedDict):
    graph_state: dict[str, str | dict[str, str | str]]

# Nodes
def node_router(state: State):
    # Retrieve the user-provided topic
    topic = state["graph_state"].get("topic", "No topic provided")
    
    # Update the graph_state with any additional information if needed
    state["graph_state"]["processed_topic"] = topic  # Example of updating graph_state

    print(f"User-provided topic: {topic}")
    return {"graph_state": state["graph_state"]}

def node_pro(state: State):
    topic = state["graph_state"]["topic"]
    pro_args = llm.invoke(f"Generate arguments in favor of: {topic}. Answer in bullet points. Max 5 words per bullet point.")
    state["graph_state"]["result"] = {"side": "pro", "arguments": pro_args}
    return {"graph_state": state["graph_state"]}

def node_contra(state: State):
    topic = state["graph_state"]["topic"]
    contra_args = llm.invoke(f"Generate arguments against: {topic}")
    state["graph_state"]["result"] = {"side": "contra", "arguments": contra_args}
    return {"graph_state": state["graph_state"]}

# Edges
def edge_pro_or_contra(state: State):
    decision = random.choice(["node_pro", "node_contra"])
    state["graph_state"]["decision"] = decision
    print(f"Routing to: {decision}")
    return decision

# Create graph
builder = StateGraph(State)
builder.add_node("node_router", node_router)
builder.add_node("node_pro", node_pro)
builder.add_node("node_contra", node_contra)

builder.add_edge(START, "node_router")
builder.add_conditional_edges("node_router", edge_pro_or_contra)
builder.add_edge("node_pro", END)
builder.add_edge("node_contra", END)

graph = builder.compile()

# Invoke the graph with a specific topic

# %%
display(Image(graph.get_graph().draw_mermaid_png()))
# %% Invokation
initial_state = {"graph_state": {"topic": "Should dogs wear clothes?"}}
result = graph.invoke(initial_state)

# %%
console.print(Markdown(result["graph_state"]['result']['arguments'].model_dump()['content']))
# %%