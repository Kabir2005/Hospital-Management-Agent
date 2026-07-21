from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from llm_provider import get_llm
from dotenv import load_dotenv
import asyncio
import os
import aiosqlite
from os import environ
from contextlib import AsyncExitStack

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from agent_state import HospitalState
from nodes.router import *
from nodes.rewrite_query import *
from nodes.info import *
from nodes.symptom_checker import *
from nodes.appointment import *
from nodes.history import *
from nodes.needs_tool import *
from nodes.fallback import *

load_dotenv()

llm = get_llm()

graph = StateGraph(HospitalState)
graph.add_node("rewrite_query", rewrite_query)
graph.add_node("router", router)
graph.add_node("info", info_node)
graph.add_node("appointment", appointment)
graph.add_node("history", history)
graph.add_node("fallback", fallback)

graph.set_entry_point("rewrite_query")
graph.add_edge("rewrite_query", "router")
graph.add_conditional_edges("symptom_checker", needs_tool)
graph.add_edge("appointment", END)

config = {"configurable": {"thread_id": 202}}

state = {
    "messages": [],
    "patient_name": "random dude3",
    "rewrite_count": 0,
    "patient_id": "1344",
}

app = None  # ✅ Initialize app globally

# Resolve the memory DB relative to this file so it works locally and in Docker
# (in the container this file lives at /app, so this still resolves to /app/databases/...).
DB_PATH = os.getenv(
    "HOSPITAL_DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "databases", "hospital_agent_memory.db"),
)

# The MCP (Tavily) subprocess and session must stay alive for the whole lifetime of the
# app, since the compiled graph holds tools bound to that session. We keep the async
# contexts open in a module-level ExitStack instead of a `with` block that would close
# them (and kill the tool connection) the moment init_app() returns.
_mcp_stack: AsyncExitStack | None = None


async def init_app():
    global app, _mcp_stack

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = await aiosqlite.connect(DB_PATH)
    memory = AsyncSqliteSaver(conn)

    from mcp.client.stdio import stdio_client, StdioServerParameters
    from mcp import ClientSession
    from langchain_mcp_adapters.tools import load_mcp_tools
    from langgraph.prebuilt.tool_node import ToolNode

    tavily_key = environ.get("TAVILY_API_KEY")
    if not tavily_key:
        raise RuntimeError(
            "TAVILY_API_KEY is not set. Add it to your .env file (see .env.example) — "
            "the symptom checker's web-search tool needs it."
        )

    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "tavily-mcp@0.1.4"],
        env={**environ, "TAVILY_API_KEY": tavily_key},
    )

    # Open the MCP stdio client + session and keep them open for the app's lifetime.
    _mcp_stack = AsyncExitStack()
    read, write = await _mcp_stack.enter_async_context(stdio_client(server_params))
    session = await _mcp_stack.enter_async_context(ClientSession(read, write))
    await session.initialize()
    tavily_tools = await load_mcp_tools(session)

    symptom_tool_node = ToolNode(tavily_tools)
    graph.add_node("symptom_tool_node", symptom_tool_node)
    graph.add_edge("symptom_tool_node", "symptom_checker")

    async def symptom_checker_node(state):
        return await symptom_checker(state, tavily_tools)

    graph.add_node("symptom_checker", symptom_checker_node)

    app = graph.compile(checkpointer=memory)
    return app  # ✅ return the compiled app


async def shutdown_app():
    """Close the long-lived MCP session/subprocess on app shutdown."""
    global _mcp_stack
    if _mcp_stack is not None:
        await _mcp_stack.aclose()
        _mcp_stack = None

# ✅ Do not run init_app directly if imported in FastAPI
if __name__ == "__main__":
    async def main():
        await init_app()
        while True:
            user_input = input("User: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting chat.")
                break

            state["messages"].append(HumanMessage(content=user_input))
            events = app.astream(state, config=config, stream_mode="values")

            async for event in events:
                last_msg = event["messages"][-1]

                if isinstance(last_msg, AIMessage):
                    content = last_msg.content
                    if isinstance(content, dict):
                        print("Assistant:", content.get("content", content))
                    else:
                        print("Assistant:", str(content).strip())
                else:
                    event["messages"][-1].pretty_print()

    # Commented out CLI run for frontend focus
    # asyncio.run(main())
