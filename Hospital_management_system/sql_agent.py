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

@tool
def get_todays_date() -> str:
    """Returns today's date in the format: 'June 17, 2025'."""
    return datetime.now().strftime("%B %d, %Y")


llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")

db = SQLDatabase.from_uri("sqlite:////app/databases/hospital.db")
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
sqltools=toolkit.get_tools()


prompt_template =""" You are an intelligent assistant with access to a hospital management database using SQLite.
You are provided with the following schema:

---

### Table: `doctors`
Stores information about doctors.
- `doctor_id` (INTEGER, Primary Key): Unique identifier for each doctor.
- `name` (TEXT): Full name of the doctor. Name is always prefixed with Dr. For example- Dr. Aman Mittal 
- `department` (TEXT): Department the doctor belongs to (e.g., Cardiology, Pediatrics, etc.). 

### Table: `appointments`
Stores patient appointment records.
- `id` (INTEGER, Primary Key, AUTOINCREMENT): Unique appointment identifier.
- `patient_id` (TEXT): Unique ID of the patient.
- `patient_name` (TEXT): Full name of the patient.
- `doctor_id` (INTEGER, Foreign Key): Refers to `doctors(doctor_id)`.
- `appointment_time` (TEXT): Scheduled date and time (e.g., "2025-06-18 10:30").

### Table: `history`
Stores past medical visits and diagnoses.
- `id` (INTEGER, Primary Key, AUTOINCREMENT): Unique history entry.
- `patient_id` (TEXT): Unique ID of the patient.
- `patient_name` (TEXT): Full name of the patient.
- `visit_date` (TEXT): Date of the visit (e.g., "2023-09-14").
- `diagnosis` (TEXT): Diagnosis made during the visit.

---

Use only the information provided in these tables to answer user queries. Follow these rules:
1. Translate the user question into a syntactically correct **SQLite SQL** query.
2. Query only the **relevant tables** and **columns**.
3. Never assume the presence of additional tables or columns.
4. If a patient's name is provided, match both `patient_id` and `patient_name` where possible.

"""
system_message = prompt_template.format(dialect="SQLite", top_k=5)
sql_agent = create_react_agent(llm, tools=sqltools+[get_todays_date], prompt=system_message)

