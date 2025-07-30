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



class Router(BaseModel):
        next: Literal["info", "appointment", "symptom_checker", "history", "rewrite_query", "fallback"]=Field(
            description="Determines which node to activate next in the workflow sequence: "
                            "info': For general hospital information (location, timings, founder, departments)"
                            "appointments': For booking or asking about appointments"
                            "symptom_checker': If the user describes a medical issue or symptom"
                            "history': If the user asks about their past visits or records"
                            "rewrite_query': If the question is vague or follow-up without clear reference"
                            "fallback': If the query doesn't match any category"
        
        )

def router(state: HospitalState) -> Command[Literal["info", "appointment", "symptom_checker", "history", "rewrite_query", "fallback"]]:
        user_query = state["messages"][-1].content
        history = "\n".join([m.content for m in state["messages"][:-1]])
        count = state.get("rewrite_count", 0)


        classification_prompt = f"""
You are an intent classification module for a hospital assistant chatbot. Your task is to analyze the user's latest message and choose **only one** most appropriate intent label based on the meaning and clarity of their message.

You must choose from the following categories:

1. info:
   Use this **only if** the query clearly asks about hospital details like:
   - Location, address
   - Timings, open hours
   - Departments or services offered
   - Founders or hospital background

2. appointment:
   The user wants to:
   - Book, cancel, or reschedule an appointment
   - Inquire about availability of doctors
   - Ask when they can meet a doctor

3. symptom_checker:
   The user is describing health-related problems or symptoms such as:
   - "I have a sore throat"
   - "I'm feeling dizzy"
   - "What should I do for chest pain?"

4. history:
   The user is referring to their own past medical data:
   - Previous treatments or diagnoses
   - Records of visits
   - Personal medical reports

5. rewrite_query:
   Use this **when the message is vague, ambiguous, or a follow-up that makes sense only with prior context**.
   Examples:
   - "And what about timings?" (without knowing what was discussed before)
   - "What about surgery?" (with no prior mention)
   - "Yes" or "Tell me more"

6. fallback:
   Use this **only if none of the above apply**, such as:
   - Completely off-topic inputs
   - Jokes, insults, or confusing gibberish

Important Rules:
- **Do NOT** default to "info" just because the query is unclear.
- If you're unsure and the message lacks specific intent, prefer "rewrite_query".
- Only respond with one of the following (no punctuation or explanation): info, appointment, symptom_checker, history, rewrite_query, fallback

Conversation so far:
{history}

User's latest message:
{user_query}

Your answer (just one label):
"""



        classification = llm.with_structured_output(Router).invoke(classification_prompt)
        print(classification)
        goto=classification.next

        if count>=3:
            return Command(
                goto="fallback"
            )
    
        return Command(
            goto=goto
        )