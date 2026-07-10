from __future__ import annotations

import os
import urllib.request
import json
from typing import Any, Optional

from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

def get_model_context_limit(provider: str, model: str) -> int:
    """Return the context token limit for the configured model."""
    provider = provider.lower().strip()
    model = model.strip()

    if provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        req = urllib.request.Request(
            f"{base_url}/api/show",
            data=json.dumps({"model": model}).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=3) as r:
                data = json.loads(r.read().decode())
                model_info = data.get("model_info", {})
                for k, v in model_info.items():
                    if "context_length" in k:
                        return int(v)
        except Exception:
            pass
        return 8192  # Safe default for local models if query fails

    if provider == "groq":
        if "llama-3.3" in model or "llama-3.1" in model:
            return 128000
        if "mixtral" in model:
            return 32768
        return 8192

    if provider == "gemini":
        return 1048576

    if provider == "openai":
        return 128000

    return 8192  # Safe generic fallback

def get_max_tokens_per_chunk(provider: str, model: str) -> int:
    """Determine the optimal chunk size based on model context limits."""
    ctx_limit = get_model_context_limit(provider, model)
    target_chunk_size = 5000
    # Keep chunk size under half the context window so that prompts + outputs fit comfortably
    return min(target_chunk_size, max(1000, ctx_limit // 2))

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
        ctx_limit = get_model_context_limit(provider, model)
        return ChatOllama(model=model, base_url=base_url, temperature=temp, num_ctx=ctx_limit, timeout=300)

    else:
        raise ValueError(
            f"Unknown provider '{provider}'. Supported: groq, gemini, openai, ollama."
        )
