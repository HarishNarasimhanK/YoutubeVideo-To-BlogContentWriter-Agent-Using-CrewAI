from __future__ import annotations

import os
from typing import Any, Optional

from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

def get_llm(provider: str, model: str, api_key: Optional[str] = None, temperature: Optional[float] = None) -> Any:
    """Return a configured LangChain ChatModel instance for the requested provider."""
    provider = provider.lower().strip()
    model = model.strip()
    temp = temperature if temperature is not None else 0.1

    if provider == "groq":
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("Groq API key required.")
        os.environ["GROQ_API_KEY"] = key
        return ChatGroq(model=model, temperature=temp, max_retries=3)

    elif provider == "gemini":
        key = (
            api_key
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )
        if not key:
            raise ValueError("Gemini API key required.")
        os.environ["GEMINI_API_KEY"] = key
        os.environ["GOOGLE_API_KEY"] = key
        return ChatGoogleGenerativeAI(model=model, temperature=temp, max_retries=3)

    elif provider == "openai":
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OpenAI API key required.")
        os.environ["OPENAI_API_KEY"] = key
        return ChatOpenAI(model=model, temperature=temp, max_retries=3)

    elif provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model=model, base_url=base_url, temperature=temp, num_ctx=8192, timeout=300)

    else:
        raise ValueError(
            f"Unknown provider '{provider}'. Supported: groq, gemini, openai, ollama."
        )
