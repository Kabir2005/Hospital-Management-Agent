from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import asyncio
import aiosqlite
from os import environ

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

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")

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

async def init_app():
    global app
    conn = await aiosqlite.connect("/app/databases/hospital_agent_memory.db")
    memory = AsyncSqliteSaver(conn)

    from mcp.client.stdio import stdio_client, StdioServerParameters
    from mcp import ClientSession
    from langchain_mcp_adapters.tools import load_mcp_tools
    from langgraph.prebuilt.tool_node import ToolNode

    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "tavily-mcp@0.1.4"],
        env={
            **environ,
            "TAVILY_API_KEY": environ["TAVILY_API_KEY"],
        },
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
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
