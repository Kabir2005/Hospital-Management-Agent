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

def fallback(state: HospitalState) -> HospitalState:
        user_message = state["messages"][-1].content.lower()
        
        rejection_phrases = [
            "i don't want", "i dont want", "not interested",
            "no appointment", "maybe later", "no thanks", "not now", "i'll book later"
        ]
        
        if any(phrase in user_message for phrase in rejection_phrases):
            polite_response = (
                "👍 No problem! If you need help booking an appointment later, just let me know.\n"
                "Is there anything else I can help you with?"
            )
        else:
            polite_response = "I'm sorry, I didn't understand that. Could you please rephrase?"

        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=polite_response)],
            "rewrite_count":0
        }