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



load_dotenv()

@tool
def get_todays_date() -> str:
    """Returns today's date in the format: 'June 17, 2025'."""
    return datetime.now().strftime("%B %d, %Y")

llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")


search_tool=TavilySearchResults()
symptom_tool_node=ToolNode([search_tool])
llm_with_search_tool=llm.bind_tools(tools=[search_tool])

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

# 1. Load the .txt file------------------RAG SETUP-------------------------------------------------------------------------------
loader = TextLoader("/app/kailash_info.txt")
documents = loader.load()

# 2. Split into chunks (important for RAG quality!)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=100
)
docs = splitter.split_documents(documents)

# 3. Embed and persist
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embedding_db = Chroma(
    persist_directory="/app/databases/Hospital_RAG_db",
    embedding_function=embedding_model
)
embedding_db.add_documents(docs)

retriever = embedding_db.as_retriever(search_type="mmr",search_kwargs={'k':7})
rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

with SqliteSaver.from_conn_string("/app/databases/hospital_agent_memory.db") as memory:

#------------------------------------------------------------------------------------------------------------------
    class HospitalState(TypedDict):
        messages: Annotated[list, add_messages]
        patient_name: Optional[str]
        rewrite_count:Optional[int]
        patient_id: Optional[str]

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
        print(rewritten)
        updated_messages = state["messages"][:-1] + [HumanMessage(content=rewritten)]

        return {
            **state,
            "messages": updated_messages,
            "rewrite_count": count + 1
        }

    

    def info_node(state: HospitalState) -> HospitalState:
        query = state["messages"][-1].content
        answer = rag_chain.run(query)
        return {**state, "messages": state["messages"] + [AIMessage(content=answer)]}
    



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
You must return your response in the following structured format:

---
🔍 **Follow-up Questions:** (Ask if any key information is missing — such as age, duration, recent travel, etc. Otherwise, write "None")

📋 **Possible Causes & Summary:** (Summarize what the symptoms may point toward, including any systems/organs possibly involved, the seriousness,and next steps)

🏥 **Suggested Department:** (Choose the most appropriate department from this list ONLY):
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
        # f"🔍 **Follow-up Questions:** {response.follow_up_questions or 'None'}\n\n"
        # f"📋 **Diagnosis Summary:** {response.diagnosis_summary}\n\n"
        # f"🏥 **Suggested Department:** {response.suggested_department}\n\n"
        # f"Would you like to book an appointment at Kailash Hospital with a {response.suggested_department} specialist?"
        # )
        
        return {
            **state,
            "messages": state["messages"] + [response],#replace response with AIMessage[content=response.content] orformatted_response
            "rewrite_count": 0
        } 


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

The user's input will always include a department.

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
                result["messages"][-1].pretty_print()

            response_msg = str(result)
        except Exception as e:
            response_msg = f"An error occurred while handling the appointment: {str(e)}"

        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=response_msg)],
            "rewrite_count":0
        }



    # Node: history
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
                result["messages"][-1].pretty_print()

            response_msg = str(result)

        except Exception as e:
            response_msg = f"An error occurred while retrieving medical history: {str(e)}"

        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=response_msg)],
            "rewrite_count":0
        }
    


    # Node: fallback
    def fallback(state: HospitalState) -> HospitalState:
        user_message = state["messages"][-1].content.lower()
        
        rejection_phrases = [
            "i don't want", "i dont want", "not interested",
            "no appointment", "maybe later", "no thanks", "not now", "i'll book later", "no"
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
    
    from langgraph.graph import StateGraph, END

    def needs_tool(state: HospitalState):
        last_msg = state["messages"][-1]
        print("TOOL CALL CHECK:", getattr(last_msg, "tool_calls", None))  # Log tool_calls
        if hasattr(last_msg, "tool_calls") and len(last_msg.tool_calls) > 0:
            return "symptom_tool_node"
        return "end"
        


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

    config = {"configurable": {"thread_id": 100}}

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
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chat.")
            break
        state["messages"].append(HumanMessage(content=user_input))
        events= app.stream(state,config=config,stream_mode="values")
        for event in events:
            event["messages"][-1].pretty_print()





