
from langchain_core.messages import HumanMessage, AIMessage
from langchain import hub
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.types import Command
from langchain_core.tools import tool
from langchain.tools import tool
from datetime import datetime
from langgraph.graph import END
from typing import Annotated, TypedDict, Optional
from langgraph.graph import StateGraph, add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from IPython.display import Image,display
from pydantic import BaseModel, ValidationError,Field
from typing import Literal, Annotated, Sequence, List
from langchain_community.tools import TavilySearchResults
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

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

llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")

search_tool=TavilySearchResults()
symptom_tool_node=ToolNode([search_tool])

conn = sqlite3.connect("/Users/kabir/Desktop/reAct_agent/Hospital_management_system/databases/hospital_agent_memory.db",check_same_thread=False)
memory = SqliteSaver(conn)


# Wiring the state graph with persistent memory
graph = StateGraph(HospitalState)
graph.add_node("rewrite_query", rewrite_query)
graph.add_node("router", router)
graph.add_node("info", info_node)
graph.add_node("symptom_tool_node",symptom_tool_node)
graph.add_node("symptom_checker", symptom_checker)
graph.add_node("appointment", appointment)
graph.add_node("history", history)
graph.add_node("fallback", fallback)

graph.set_entry_point("rewrite_query")
graph.add_edge("rewrite_query", "router")
graph.add_conditional_edges("symptom_checker",needs_tool)
graph.add_edge("symptom_tool_node","symptom_checker")
graph.add_edge("appointment",END)

app = graph.compile(checkpointer=memory)

config = {"configurable": {"thread_id": 200}}

# png_data = app.get_graph(xray=True).draw_mermaid_png()
# with open("hospital_agent_graph.png", "wb") as f:
#   f.write(png_data)

# #macOS-specific: opens the image
# import os
# os.system("open hospital_agent_graph.png")

state = {
    "messages": [],
    "patient_name": "random dude3",
    "rewrite_count": 0,
    "patient_id": "1344",
}


#result=app.invoke(state,config=config)
#print(result)
if __name__ == "__main__":

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chat.")
            break

        state["messages"].append(HumanMessage(content=user_input))
        events = app.stream(state, config=config, stream_mode="values")

        for event in events:
            last_msg = event["messages"][-1]


            # Check if it's an AIMessage and if its content is a dictionary-like string or object
            if isinstance(last_msg, AIMessage):
                content = last_msg.content
                if isinstance(content, dict):
                    print("Assistant:", content.get("content", content))
                elif isinstance(content, str) and content.strip().startswith("{") and "messages" in content:
                    # Heuristic: Looks like a dict string dump with messages
                    try:
                        import json
                        parsed = json.loads(content)
                        if isinstance(parsed, dict):
                            print("Assistant:", parsed.get("content", parsed))
                        else:
                            print("Assistant:", content.strip())
                    except Exception:
                        print("Assistant:", content.strip())
                else:
                    # Normal output
                    print("Assistant:", content.strip())
            else:
                # Fallback: pretty print if not AIMessage or unknown
                event["messages"][-1].pretty_print()