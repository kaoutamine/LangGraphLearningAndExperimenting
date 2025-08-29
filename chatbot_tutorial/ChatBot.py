from typing import Annotated

from typing_extensions import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from IPython.display import Image, display

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt

from langchain_tavily import TavilySearch
from langchain_core.tools import tool  # Add this import!
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

print("hello")
#initiate the llm
llm = init_chat_model("openai:gpt-4.1")

#initiate the tools (search functionality)
tool_search = TavilySearch(max_results=2)

@tool
def human_assistance(query: str) -> str:
    """Call this when you need human assistance or expert guidance."""
    human_response = interrupt({"query": query})
    return human_response["data"]

tools = [tool_search, human_assistance]

#initiate the memory checkpointer
memory = InMemorySaver()

llm_with_tools = llm.bind_tools(tools)

graph_builder = StateGraph(State)

#add chatbot node and end/start
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)


tool_node = BasicToolNode(tools=tools)

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
graph = graph_builder.compile(checkpointer=memory)

try:
    # Save the graph as a PNG file instead of trying to display it
    graph_png = graph.get_graph().draw_mermaid_png()
    with open("chatbot_graph.png", "wb") as f:
        f.write(graph_png)
    print("Graph saved as 'chatbot_graph.png'")
except Exception as e:
    print(f"Could not generate graph image: {e}")
    # This requires some extra dependencies and is optional



print("starting the streaming process")
user_input = "I need some expert guidance for building an AI agent. Could you request assistance for me?"
config = {"configurable": {"thread_id": "1"}}

events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()



human_response = (
    "We, the experts are here to help! We'd recommend you check out LangGraph to build your agent."
    " It's much more reliable and extensible than simple autonomous agents."
)

human_command = Command(resume={"data": human_response})

events = graph.stream(human_command, config, stream_mode="values")
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()