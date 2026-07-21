"""Central LLM factory so every node shares one provider/model configuration.

Uses Groq (fast, generous free tier). Two models by default:
  * reasoning nodes (routing, info/RAG, symptom triage) -> a strong 70B model
  * the SQL tool agent -> a smaller model that is more reliable at tool calling
Both are overridable via GROQ_MODEL / GROQ_SQL_MODEL env vars.
"""
import os

from langchain_groq import ChatGroq

DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_SQL_MODEL = "llama-3.1-8b-instant"


def get_llm(model: str | None = None):
    return ChatGroq(
        model=model or os.getenv("GROQ_MODEL", DEFAULT_MODEL),
        temperature=0,
    )
