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
load_dotenv()

llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")

def rewrite_query(state: HospitalState) -> HospitalState:
        count = state.get("rewrite_count", 0)

        # Abort loop if we've rewritten too many times
        if count >= 3:
            return {
                **state,
                "messages": state["messages"] + [AIMessage(content="I'm having trouble understanding your query. Could you please clarify?")],
                "rewrite_count": count  # Don't increment further
            }

        history = "\n".join([m.content for m in state["messages"][:-1]])
        current = state["messages"][-1].content
        prompt = f"""You are a helpful assistant designed to improve user queries for a hospital chatbot (Kailash Hospital). 
        Your goal is to rewrite the user's question to be fully self-contained, unambiguous, and contextually complete.
        Use the conversation history to infer missing information such as symptoms, intent (e.g., appointment, information, diagnosis), and relevant details (e.g., department, age, gender, urgency).

Conversation History:
{history}

Original Question:
{current}

Rewrite the above question to include any inferred or implied context so it can be understood independently, even without the history. Be concise but informative.

Rewritten Query:"""

        rewritten = llm.invoke(prompt).content
        #print(rewritten)
        updated_messages = state["messages"][:-1] + [HumanMessage(content=rewritten)]

        return {
            **state,
            "messages": updated_messages,
            "rewrite_count": count + 1
        }

