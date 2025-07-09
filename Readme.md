# 🏥 Kailash Hospital AI Agent

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-async-green?logo=fastapi)
![LangChain](https://img.shields.io/badge/LangChain-RAG-yellow?logo=langchain)
![SQLite](https://img.shields.io/badge/SQLite-DB-lightgrey?logo=sqlite)

A full-stack conversational hospital assistant powered by Gemini, LangGraph, Retrieval-Augmented Generation, and TavilyMCP — complete with FastAPI backend, TailwindCSS web UI, and SQL-driven memory and appointment management.

---

## 🚀 Overview
Kailash Hospital AI Agent is an intelligent, full-stack hospital assistant designed to streamline patient interaction, symptom triage, and appointment workflows via natural conversation. It offers a rich chat interface powered by Gemini (gemini-2.0-flash-001) and a modular backend built using LangGraph, LangChain, FastAPI, SQL (SQLite), and TavilyMCP for advanced search.

**Key Capabilities:**
- 🔍 Answer factual queries about Kailash Hospital (departments, timings, services, etc.)
- 🩺 Perform smart symptom checking with triage suggestions and department routing
- 📅 Handle appointment workflows (viewing, scheduling, updating) using SQL logic
- 🧠 Track memory, patient state, and chat history persistently via SQLite
- 💬 Serve conversations through a web-based chat UI styled with Tailwind CSS
- 🌐 Use TavilyMCP for real-time, high-quality web search in symptom triage and information retrieval

---

## ✨ Features
- **FastAPI-powered Backend:** Robust, asynchronous API layer for the hospital agent.
- **Gemini LLM + LangGraph State Machine:** Structured, node-based reasoning over user messages.
- **Retrieval-Augmented Generation (RAG):** Answers grounded in official Kailash Hospital knowledge base using HuggingFace embeddings + Chroma.
- **SQL-Based Logic:** Appointment workflows and persistent memory via SQLite.
- **TavilyMCP Integration:** Uses TavilyMCP (via MCP protocol and Node.js) for advanced, real-time web search in medical triage and information flows.
- **Async Tool Loading:** TavilyMCP tools are loaded asynchronously and bound to the LLM at runtime for dynamic, up-to-date search.
- **Beautiful Tailwind UI:** Minimal, responsive HTML+Tailwind interface.

---

## 🛠️ Tech Stack
| Layer      | Tech                                                                 |
|------------|----------------------------------------------------------------------|
| UI         | HTML5, Tailwind CSS                                                 |
| Backend    | FastAPI (Python)                                                    |
| LLM        | Gemini (gemini-2.0-flash-001) via ChatGoogleGenerativeAI            |
| Orchestration | LangGraph                                                        |
| RAG        | RetrievalQA, Chroma, HuggingFace (MiniLM-L6-v2)                     |
| Memory     | LangGraph SqliteSaver (persistent, thread-based)                    |
| Database   | SQLite + SQL logic for appointments and patient history             |
| Tools      | TavilyMCP (via MCP protocol), LangChain SQL agent                   |

---

## 🖥️ Architecture

![Hospital Agent Graph](hospital_agent_graph.png)

```
UI (Tailwind HTML) → FastAPI → LangGraph → Router
  ├─ InfoNode → RAGChain → ChromaDB
  ├─ SymptomChecker → TavilyMCP ToolNode
  └─ Appointment → SQLAgent → SQLite
```

---

## 🏁 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourname/hospital-agent.git
cd hospital-agent/Hospital_management_system
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Node.js and TavilyMCP
- **Node.js** is required for TavilyMCP. [Download Node.js](https://nodejs.org/)
- TavilyMCP is run via `npx` (no global install needed):
  - The backend will launch TavilyMCP using:
    ```bash
    npx -y tavily-mcp@0.1.4
    ```
- **Tavily API Key:**
  - Get your API key from [Tavily](https://app.tavily.com/).
  - Add it to your `.env` file:
    ```env
    TAVILY_API_KEY=your-tavily-api-key
    ```

### 4. Environment Setup
- Add your Gemini API key to `.env`:
  ```env
  GEMINI_API_KEY=your-gemini-api-key
  ```
- Prepare `kailash_info.txt` with hospital data (sample provided).
- Set up `hospital.db` (SQLite) with the correct schema for appointments/patients.

### 5. Run the FastAPI Server
```bash
uvicorn api_setup:app --reload
```

### 6. Access the UI
Open your browser at [http://localhost:8000](http://localhost:8000)

---

## 💬 Example Interactions

```
User: I have mild chest pain and shortness of breath.
Assistant: These symptoms may indicate a cardiac issue. (Uses TavilyMCP to search for latest guidelines.) I recommend consulting the Cardiology department. Would you like help accessing appointment options?
```

---

## 📂 Project Structure

```
Hospital_management_system/
├── agent_runnable.py         # Main agent logic
├── api_setup.py              # FastAPI app
├── sql_agent.py              # SQL agent for appointments
├── nodes/                    # LangGraph nodes
├── kailash_info_store/       # Info retrieval logic
├── databases/                # SQLite DB and helpers
├── kailash_info.txt          # Hospital info knowledge base
├── chatbot.html              # Web UI
├── hospital_agent_graph.png  # Architecture diagram
└── ...
```

---

## 🧩 TavilyMCP Integration
- **What is TavilyMCP?**
  - TavilyMCP is a Node.js-based MCP (Machine Control Protocol) server that provides high-quality, real-time web search as a tool for LLM agents.
- **How is it used?**
  - The backend launches TavilyMCP via `npx` and connects using the `mcp` and `langchain-mcp-adapters` Python packages.
  - The symptom checker and other nodes can invoke TavilyMCP tools asynchronously for up-to-date information.
- **Setup:**
  - Requires Node.js and a Tavily API key in your `.env` file.
  - No manual server start needed; the backend handles launching TavilyMCP as needed.

---

## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License
[MIT](LICENSE)

---

**Kailash Hospital AI Agent** — Smart, conversational healthcare for everyone.
