# 🏥 Kailash Hospital AI Agent

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-async-green?logo=fastapi)
![LangGraph](https://img.shields.io/badge/LangGraph-graph--based-blueviolet?logo=python)
![Groq](https://img.shields.io/badge/Groq-LLM-orange)
![SQLite](https://img.shields.io/badge/SQLite-DB-blue?logo=sqlite)

A full-stack conversational hospital assistant powered by Groq (Llama 3.3), LangGraph, Retrieval-Augmented Generation, and MCP (Model Context Protocol) server-based tool calling — complete with FastAPI backend, TailwindCSS web UI, and SQLite-driven memory and appointment management.

---

## 🚀 Overview
Kailash Hospital AI Agent is an intelligent, full-stack hospital assistant designed to streamline patient interaction, symptom triage, and appointment workflows via natural conversation. It offers a rich chat interface powered by Groq (Llama 3.3 70B) and a modular backend built using LangGraph, FastAPI, SQLite, and MCP (Model Context Protocol) for advanced tool integration.

**Key Capabilities:**
- 🔍 Answer factual queries about Kailash Hospital (departments, timings, services, etc.)
- 🩺 Perform smart symptom checking with triage suggestions and department routing
- 📅 Handle appointment workflows (viewing, scheduling, updating) using SQLite logic
- 🧠 Track memory, patient state, and chat history persistently via SQLite
- 💬 Serve conversations through a web-based chat UI styled with Tailwind CSS
- 🌐 Use MCP server-based tools (TavilyMCP) for real-time, high-quality web search in symptom triage and information retrieval

---

## ✨ Features
- **FastAPI-powered Backend:** Robust, asynchronous API layer for the hospital agent.
- **Groq LLM + LangGraph State Machine:** Structured, node-based reasoning over user messages.
- **Retrieval-Augmented Generation (RAG):** Answers grounded in official Kailash Hospital knowledge base using HuggingFace embeddings + Chroma.
- **SQLite-Based Logic:** Appointment workflows and persistent memory via SQLite.
- **MCP Server Integration:** Uses MCP (Model Context Protocol) for modular tool integration, including TavilyMCP for advanced, real-time web search in medical triage and information flows.
- **Modular Tooling:** Easily extend the agent with new tools and capabilities via the MCP protocol.
- **Beautiful Tailwind UI:** Minimal, responsive HTML+Tailwind interface.

---

## 🛠️ Tech Stack
| Layer      | Tech                                                                 |
|------------|----------------------------------------------------------------------|
| UI         | HTML5, Tailwind CSS                                                 |
| Backend    | FastAPI (Python)                                                    |
| LLM        | Groq — Llama 3.3 70B (reasoning) + Llama 3.1 8B (SQL tools) via langchain-groq |
| Orchestration | LangGraph                                                        |
| RAG        | RetrievalQA, Chroma, HuggingFace (MiniLM-L6-v2)                     |
| Memory     | LangGraph SqliteSaver (persistent, thread-based)                    |
| Database   | SQLite + SQL logic for appointments and patient history         |
| Tools      | MCP (Model Context Protocol), TavilyMCP, LangGraph SQL agent        |

---

## 🖥️ Architecture

![Hospital Agent Graph](Hospital_management_system/hospital_agent_graph.png)

```
UI (Tailwind HTML) → FastAPI → LangGraph → Router
  ├─ InfoNode → RAGChain → ChromaDB
  ├─ SymptomChecker → MCP ToolNode (TavilyMCP)
  └─ Appointment → SQLAgent → SQLite
```

---

## 🏁 Getting Started

### Option 1: Using Docker (Recommended)

#### 1. Clone the Repository
```bash
git clone https://github.com/Kabir2005/Hospital-Management-Agent.git
cd Hospital-Management-Agent/Hospital_management_system
```

#### 2. Environment Setup
- Create a `.env` file with your API keys:
  ```env
  GROQ_API_KEY=your-groq-api-key
  TAVILY_API_KEY=your-tavily-api-key
  ```

#### 3. Run with Docker Compose
```bash
docker-compose up -d
```

#### 4. Access the UI
Open your browser at [http://localhost:8000](http://localhost:8000)

### Option 2: Manual Setup

#### 1. Clone the Repository
```bash
git clone https://github.com/Kabir2005/Hospital-Management-Agent.git
cd Hospital-Management-Agent/Hospital_management_system
```

#### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

#### 3. Install Node.js and MCP Tools
- **Node.js** is required for MCP servers. [Download Node.js](https://nodejs.org/)
- MCP tools are run via `npx` (no global install needed):
  - The backend will launch MCP servers using:
    ```bash
    npx -y tavily-mcp@0.1.4
    ```
- **Tavily API Key:**
  - Get your API key from [Tavily](https://app.tavily.com/).
  - Add it to your `.env` file:
    ```env
    TAVILY_API_KEY=your-tavily-api-key
    ```

#### 4. Environment Setup
- Copy `.env.example` to `.env` and fill in your keys:
  ```env
  GROQ_API_KEY=your-groq-api-key
  TAVILY_API_KEY=your-tavily-api-key
  ```
- `kailash_info.txt` (the RAG knowledge base) is included.

#### 5. Seed the SQLite database
Creates `databases/hospital.db` with sample doctors, appointments, and patient history:
```bash
cd databases && python db_setup.py && cd ..
```

#### 6. Run the FastAPI Server
```bash
uvicorn api_setup:app_fastapi --host 0.0.0.0 --port 8000
```

#### 7. Access the UI
Open your browser at [http://localhost:8000](http://localhost:8000)

---

## 💬 Example Interactions

```
User: I have mild chest pain and shortness of breath.
Assistant: These symptoms may indicate a cardiac issue. (Uses MCP tools to search for latest guidelines.) I recommend consulting the Cardiology department. Would you like help accessing appointment options?
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

## 🧩 MCP Server Integration
- **What is MCP?**
  - MCP (Model Context Protocol) is a protocol for connecting AI agents to external tools and data sources.
  - It enables dynamic, modular tool integration through standardized server interfaces.
- **How is it used?**
  - The backend launches MCP servers (like TavilyMCP) via `npx` and connects using the `mcp` and `langchain-mcp-adapters` Python packages.
  - The symptom checker and other nodes can invoke MCP tools for up-to-date information and external functionality.
- **Current MCP Tools:**
  - **TavilyMCP:** Provides high-quality, real-time web search for medical information and symptom analysis.
- **Setup:**
  - Requires Node.js and appropriate API keys in your `.env` file.
  - No manual server start needed; the backend handles launching MCP servers as needed.

---

## 🚂 Deploy to Railway

This app is a long-running server (FastAPI + a Node/Tavily MCP subprocess + SQLite), so it
belongs on a container host, **not** a serverless platform like Vercel. Railway builds the
included `Dockerfile` directly.

1. Push this repo to GitHub (already done).
2. On [railway.app](https://railway.app): **New Project → Deploy from GitHub repo** → pick
   `Hospital-Management-Agent`.
3. **Set the Root Directory** — this app's `Dockerfile` lives in the `Hospital_management_system/`
   subfolder, so in the service's **Settings → Root Directory** enter:
   ```
   Hospital_management_system
   ```
   (Otherwise Railway won't find the Dockerfile.)
4. Add environment variables under **Variables**:
   ```
   GROQ_API_KEY=your-groq-key
   TAVILY_API_KEY=your-tavily-key
   ```
   (Optional: `GROQ_MODEL`, `GROQ_SQL_MODEL` to override the defaults.)
5. Railway builds the image and starts it. The container binds to Railway's injected `$PORT`
   automatically. Under **Settings → Networking → Generate Domain** to get a public URL.
6. Open `https://<your-app>.up.railway.app/chatbot.html` for the chat UI.

### What's inside the container (your DB question)

The image is fully self-contained:

- **Python + all deps** and **Node.js + Tavily MCP** (installed at build time).
- The **SQLite database** (`hospital.db`) — the entrypoint runs `db_setup.py` on first boot,
  so doctors/appointments/history are seeded automatically.
- The **Chroma RAG store** — rebuilt from `kailash_info.txt` on startup.

So you don't provision a separate database; SQLite is a file inside the container.

⚠️ **One caveat:** Railway's container filesystem is **ephemeral** — on every redeploy/restart
the SQLite files reset (the DB is re-seeded fresh, and any appointments booked at runtime or
chat memory are lost). For a demo that's fine. To **persist** data across deploys, add a
**Railway Volume** mounted at `/app/databases`.

> First cold start takes ~30–60s because the MiniLM embedding model downloads from Hugging
> Face once; subsequent restarts are faster.

---

## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License
[MIT](LICENSE)

---

**Kailash Hospital AI Agent** — Smart, conversational healthcare for everyone.