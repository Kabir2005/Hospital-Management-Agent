import os
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

import asyncio
from mcp import ClientSession
from mcp.client.stdio import stdio_client
from mcp.client.stdio import StdioServerParameters
from langchain_mcp_adapters.tools import load_mcp_tools

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Optional, Literal


llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")

# ✅ Function to load Tavily MCP tool via stdio
async def load_tavily_mcp_tool():
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "tavily-mcp@0.1.4"],
        env={
            **os.environ,  # inherit current env
            "TAVILY_API_KEY": os.getenv("TAVILY_API_KEY")  # explicitly set your key
        },
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            return tools

class SymptomCheckerResponse(BaseModel):
    follow_up_questions: Optional[str] = Field(
        default=None,
        description="Any additional questions needed for diagnosis, such as age, gender, duration of symptoms. If none, return None."
    )
    diagnosis_summary: str = Field(
        ...,
        description="Explanation of possible causes, seriousness (mild/moderate/severe), and next steps (e.g., rest, medication, see a doctor)."
    )
    suggested_department: Literal[
        "Cardiology",
        "Neurology & Neurosurgery",
        "Orthopedics",
        "Gynecology & Obstetrics",
        "Pediatrics",
        "ENT",
        "Dermatology",
        "Gastroenterology",
        "Urology",
        "Psychiatry & Mental Health"
    ] = Field(
        ...,
        description="The most appropriate hospital department the patient should visit for further evaluation."
    )

# ✅ Updated symptom_checker function to load MCP tools and bind them
async def symptom_checker(state: HospitalState, tavily_tools) -> HospitalState:
    query = state["messages"][-1].content

    system_prompt = """
You are a medically knowledgeable assistant helping users understand their symptoms.

Your goal is to clearly and informatively explain:
1. What the symptoms could possibly indicate (a few possible causes or systems involved),
2. Whether the situation might be serious or not,
3. What next steps to take — such as rest, over-the-counter medication, or seeing a specialist.

If the symptoms are vague, uncommon, complex, or could point to more than one cause, you must use the search tool-'tavily_search_results_json]` to gather reliable and up-to-date information before replying.

**Avoid giving vague or generic suggestions. Do not say "could be many things" or "you should see a doctor" without first using the `search_tool` to find plausible causes.**

Be empathetic, concise, and clear — but also informative. Do **not** make a formal medical diagnosis.
You must return your response in the following format:

---
🔍 Follow-up Questions-/n
(Ask if any key information is missing — such as age, duration, recent travel, etc. Otherwise, write "None")

📋 Possible Causes & Summary: /n
(Summarize what the symptoms may point toward, including any systems/organs possibly involved, the seriousness,and next steps)

🏥 Suggested Department: /n
(Choose the most appropriate department from this list ONLY):
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

---
Rule-do not suggest or mention a department not present in the above list.

End your message with:

"Would you like to book an appointment with a (Suggested Department) specialist?"
"""

    # ✅ Load Tavily MCP tools dynamically
    # tavily_tools = await load_tavily_mcp_tool()

    # ✅ Bind Gemini with Tavily MCP tools
    llm_with_tavily_mcp = llm.bind_tools(tools=tavily_tools)

    full_prompt = f"{system_prompt}\n\nSymptom: {query}\n\nResponse:"

    # ✅ Invoke the LLM with MCP tool binding
    response = await llm_with_tavily_mcp.ainvoke(full_prompt)

    return {
        **state,
        "messages": state["messages"] + [response],
        "rewrite_count": 0
    }

# # ✅ For standalone testing
# if __name__ == "__main__":
#     sample_state = {
#         "messages": [{"content": "I have chest pain and shortness of breath"}],
#         "rewrite_count": 0
#     }
#     asyncio.run(symptom_checker(sample_state))