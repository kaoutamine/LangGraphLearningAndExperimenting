from typing import Annotated

from typing_extensions import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from IPython.display import Image, display

import os
from langchain.chat_models import init_chat_model

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


llm = init_chat_model("openai:gpt-4.1")

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)  # You need to add the node first!
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

try:
    # Save the graph as a PNG file instead of trying to display it
    graph_png = graph.get_graph().draw_mermaid_png()
    with open("chatbot_graph.png", "wb") as f:
        f.write(graph_png)
    print("Graph saved as 'chatbot_graph.png'")
except Exception as e:
    print(f"Could not generate graph image: {e}")
    # This requires some extra dependencies and is optional
