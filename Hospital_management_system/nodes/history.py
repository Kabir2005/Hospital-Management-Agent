
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
from sql_agent import sql_agent
load_dotenv()

llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")


def history(state: HospitalState) -> HospitalState:
        # Extract patient info
        patient_id = state.get("patient_id", "unknown")

        # Construct a prompt for the SQL agent
        query = f"""
        You are a medical records assistant. Fetch and summarize the visit history of the patient with the following ID:

        - Patient ID: {patient_id}

        Task:
        - Retrieve all records from the `history` table.
        - Display the visit date and diagnosis in chronological order.
        - If no records are found, respond with "No medical history found."

        Format the output in a clear and readable list.
        """

        try:
            # Stream the response from the SQL agent
            results = sql_agent.stream({"messages": [query]}, stream_mode="values")

            for result in results:
                #result["messages"][-1].pretty_print()------print statement for debugging
                pass

            response_msg = result["messages"][-1].content

        except Exception as e:
            response_msg = f"An error occurred while retrieving medical history: {str(e)}"

        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=response_msg)],
            "rewrite_count":0
        }