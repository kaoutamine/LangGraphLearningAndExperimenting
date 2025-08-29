from typing import Annotated

from typing_extensions import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from IPython.display import Image, display

from langchain_tavily import TavilySearch
from BasicToolNode import BasicToolNode
import os
from langchain.chat_models import init_chat_model

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


load_dotenv()

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

def route_tools(state: State):
    """
    Route the tools based on the state (either END or tools)
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END



llm = init_chat_model("openai:gpt-4.1")

tool = TavilySearch(max_results=2)
tools = [tool]

llm_with_tools = llm.bind_tools(tools)

graph_builder = StateGraph(State)

#add chatbot node and end/start
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)


tool_node = BasicToolNode(tools=[tool])

graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
    # It defaults to the identity function, but if you
    # want to use a node named something else apart from "tools",
    # You can update the value of the dictionary to something else
    # e.g., "tools": "my_tools"
    {"tools": "tools", END: END},
)
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
