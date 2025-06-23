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

search_tool=TavilySearchResults()
llm_with_search_tool=llm.bind_tools(tools=[search_tool])


class SymptomCheckerResponse(BaseModel):
        follow_up_questions: Optional[str] = Field(
            default=None,
            description="Any additional questions needed for diagnosis, such as age, gender, duration of symptoms. If none, return " 
            "None."
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
            description="The most appropriate hospital department the patient should visit for further evaluation, "
        "based on the presented symptoms. Choose one from the predefined list of actual departments "
        "at Kailash Hospital. This field must align with medically relevant specialties such as ENT "
        "for ear infections, Cardiology for chest pain, or Dermatology for skin issues."
        )


def symptom_checker(state: HospitalState) -> HospitalState:
        query = state["messages"][-1].content

        system_prompt = """
You are a medically knowledgeable assistant helping users understand their symptoms.

Your goal is to clearly and informatively explain:
1. What the symptoms could possibly indicate (a few possible causes or systems involved),
2. Whether the situation might be serious or not,
3. What next steps to take — such as rest, over-the-counter medication, or seeing a specialist.

If the symptoms are vague, uncommon, complex, or could point to more than one cause, you must use the `search_tool` to gather reliable and up-to-date information before replying.

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


        
        full_prompt = f"{system_prompt}\n\nSymptom: {query}\n\nResponse:"
        response=llm_with_search_tool.invoke(full_prompt)
        # response = llm_with_search_tool.with_structured_output(SymptomCheckerResponse).invoke(full_prompt)
        # formatted_response =(
        # f"🔍 **Follow-up Questions:* {response.follow_up_questions or 'None'}\n\n"
        # f"📋 **Diagnosis Summary:** {response.diagnosis_summary}\n\n"
        # f"🏥 **Suggested Department:** {response.suggested_department}\n\n"
        # f"Would you like to book an appointment at Kailash Hospital with a {response.suggested_department} specialist?"
        # )
        
        return {
            **state,
            "messages": state["messages"] + [response],#replace response with AIMessage[content=response.content] orformatted_response
            "rewrite_count": 0
        }