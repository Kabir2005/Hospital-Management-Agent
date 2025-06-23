
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
# Node: appointment
def appointment(state: HospitalState) -> HospitalState:
        # Extract relevant patient info
        patient_id = state.get("patient_id", "unknown")
        patient_name = state.get("patient_name", "unknown")
        messages = state.get("messages", [])
        
        # Extract the last user message (symptom, intent, etc.)
        last_user_msg = messages[-1].content if messages else ""

        # Create detailed input for the SQL agent
        query = f"""
You are a hospital assistant agent responsible for managing doctor appointments using a SQLite database with three tables: `doctors`, `appointments`, and `history`.

The patient details are:
- Patient ID: {patient_id}
- Patient Name: {patient_name}

The user's input will include a department. If not included, infer from the users message.

### Department Names- 
- Cardiology
- Neurology & Neurosurgery
- Orthopedics
- Gynecology & Obstetrics
- Pediatrics
- ENT 
- Dermatology
- Gastroenterology
- Urology
- Psychiatry & Mental Health



You MUST perform **exactly one action** based on the message. Default to booking or checking availability whenever enough information is available. Only ask follow-up questions if a required field is completely missing and cannot be inferred.

---

###  Booking Rules

**Case 1: Appointment with no doctor/time**
- Choose any doctor from the department (use first row in `doctors` table).
- Find the earliest available hourly slot (09:00 to 17:00) from today using the `appointments` table.
- Book and INSERT the appointment using that doctor.

**Case 2: Appointment with doctor and/or time**
- Get `doctor_id` from name using `doctors` table.
- If time is given:
  - If doctor is available at that time → book.
  - If not → return message: "Doctor not available at that time. Available at: [next 2 free slots]."
- If time not given:
  - Book next free slot for that doctor.

**Case 3: Cancel or Reschedule**
- Identify appointment using patient name and doctor/time.
- Cancel: DELETE from `appointments`.
- Reschedule: DELETE old + INSERT new. If no time given, pick next available.

**Case 4: Doctor availability query**
- Examples: 
    - “Is Dr. Mahesh Sharma available at 3 PM?”
    - “When can I meet Dr. Asha Mehta?”
- Action:
    - Look up doctor’s `doctor_id` from the `doctors` table.
    - If time is mentioned:
        - Check if doctor is free at that time → respond yes/no.
        - If not free, suggest next 2 available slots.
    - If time is not mentioned:
        - Return next 3 available future hourly slots (within 09:00–17:00) starting today.
---

###  Constraints:
- All time values must be `YYYY-MM-DD HH:MM`, future-dated, and within 09:00–17:00.
- Prefer actions over asking unless blocked.
- If you can book, and show confirmation.
- Never explain your SQL logic unless explicitly asked.
- Never show doctor_id. Always refer by name.

---

###  User Message:
"{last_user_msg}"

What is your final response? Reply with a confirmation or a **single clarifying question** only if necessary.
"""


        try:
            # Invoke the SQL agent
            results = sql_agent.stream({"messages": [query]},stream_mode="values")
            for result in results:
                #result["messages"][-1].pretty_print()-------To print stream for debugging 
                pass

            response_msg = result["messages"][-1].content     #This returns an AI message.
        except Exception as e:
            response_msg = f"An error occurred while handling the appointment: {str(e)}"

        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=response_msg)],   
            "rewrite_count":0
        }
