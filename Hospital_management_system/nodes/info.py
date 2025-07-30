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
from langchain.prompts import PromptTemplate


from agent_state import HospitalState
load_dotenv()

# Load Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")

# ------------------ RAG SETUP ------------------

# 1. Load Kailash Hospital info
loader = TextLoader("/app/kailash_info.txt")
documents = loader.load()

# 2. Split text into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
docs = splitter.split_documents(documents)

# 3. Initialize embedding model and Chroma vectorstore
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embedding_db = Chroma(
    persist_directory="/app/databases/Hospital_RAG_db",
    embedding_function=embedding_model
)

# 4. Add docs to vectorstore (optional if already persisted)
embedding_db.add_documents(docs)

# 5. Configure retriever
retriever = embedding_db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 10}
)

# ------------------ Prompt Template ------------------

kailash_prompt = PromptTemplate.from_template(
    """You are a knowledgeable and polite assistant for **Kailash Hospital**, a multi-specialty healthcare center in India. 
Your job is to answer user questions related to the hospital's services, facilities, departments, doctors, timings, and general patient information.

Use only the **provided context** to answer the question. Do not make up information. If the answer is not found in the context, respond clearly that you don’t have that information.

Your tone should be helpful, concise, and professional, while sounding warm and reassuring.

---
**Context:**  
{context}

**User Question:**  
{question}

**Answer:**
"""
)

# 6. Build RetrievalQA RAG chain with prompt
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": kailash_prompt},
    return_source_documents=False
)

# ------------------ info_node ------------------

def info_node(state: HospitalState) -> HospitalState:
    query = state["messages"][-1].content
    answer = rag_chain.run(query)
    return {**state, "messages": state["messages"] + [AIMessage(content=answer)]}