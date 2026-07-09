import json
import os
import urllib.request
import streamlit as st

from pipeline import run_pipeline, translate_text

def get_ollama_models() -> list[str]:
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=2) as response:
            data = json.loads(response.read().decode())
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []

st.set_page_config(
    page_title="DemystifyAI — LangGraph",
    page_icon="🎥",
    layout="wide",
)

st.markdown(
    """
    <style>
        .reportview-container { background: #0f172a; }
        h1 { color: #f8fafc; font-weight: 800 !important; }
        .stTabs [data-baseweb="tab-list"] { gap: 24px; }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            background-color: #1e293b;
            border-radius: 4px;
            color: #94a3b8;
            font-weight: 600;
            padding: 10px 20px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #3b82f6;
            color: white;
        }
        .guardrail-msg {
            background: #1e293b;
            border-left: 4px solid #f59e0b;
            padding: 1.2rem 1.5rem;
            border-radius: 8px;
            color: #f1f5f9;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎥 DemystifyAI Agent")
st.markdown(
    "Converts any YouTube video into **Beginner** and **Technical** explanations "
    "with Wikipedia & arXiv citations — powered by **LangGraph** multi-agent intelligence."
)

st.sidebar.header("⚙️ Configuration")

provider = st.sidebar.selectbox(
    "Select LLM Provider",
    ["Groq", "Gemini", "OpenAI", "Ollama"],
)

if provider == "Groq":
    model = st.sidebar.selectbox(
        "Model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
    )
    key_placeholder = "Leave empty to use built-in sandbox Groq key"
elif provider == "Gemini":
    model = st.sidebar.selectbox(
        "Model",
        ["gemini-2.0-flash", "gemini-2.5-flash"],
    )
    key_placeholder = "Leave empty to use built-in sandbox Gemini key"
elif provider == "OpenAI":
    model = st.sidebar.selectbox("Model", ["gpt-4o-mini", "gpt-4o"])
    key_placeholder = "Enter your OpenAI API key"
else:
    ollama_models = get_ollama_models()
    if not ollama_models:
        st.sidebar.error("❌ Ollama server unreachable or has no models. Start Ollama and pull a model first.")
        model = ""
    else:
        model = st.sidebar.selectbox("Model", ollama_models)
    key_placeholder = "Not required for local Ollama"

custom_api_key = ""
if provider != "Ollama":
    custom_api_key = st.sidebar.text_input(
        f"{provider} API Key",
        type="password",
        placeholder=key_placeholder,
    )


st.sidebar.markdown("---")
st.sidebar.info(
    "**Pipeline v2 — LangGraph:**\n"
    "1. Extract transcript (no LLM).\n"
    "2. **Guardrail** — rejects non-educational videos.\n"
    "3. Researcher fetches Wikipedia + arXiv citations.\n"
    "4. Writers run **in parallel** (Beginner & Technical).\n"
    "5. Verifiers run **in parallel** & emit verdict.\n"
    "6. If gaps found → loops back to researcher (max 3×).\n"
    "7. Files saved & displayed here."
)

youtube_url = st.text_input(
    "🔗 Enter YouTube Video URL",
    placeholder="https://www.youtube.com/watch?v=...",
)

if "generated" not in st.session_state:
    st.session_state.generated = False
    st.session_state.explanation_text = ""
    st.session_state.rejected = False
    st.session_state.rejection_msg = ""
    st.session_state.iteration_count = 0

run_btn = st.button("🚀 Demystify Video", type="primary", use_container_width=True)

if run_btn:
    if not youtube_url.strip():
        st.error("Please enter a valid YouTube URL.")
    elif provider.lower() == "ollama" and not model:
        st.error("Please start local Ollama and pull a model first (e.g. `ollama pull qwen2.5:0.5b`).")
    else:
        with st.spinner("⏳ Running LangGraph pipeline …"):
            try:
                api_key_to_use = custom_api_key.strip() or None

                for key in ["generated", "rejected"]:
                    st.session_state[key] = False
                for key in ["explanation_text", "rejection_msg"]:
                    st.session_state[key] = ""
                st.session_state.iteration_count = 0

                for fname in ["demystified_explanation.md"]:
                    if os.path.exists(fname):
                        try:
                            os.remove(fname)
                        except OSError:
                            pass

                result = run_pipeline(
                    youtube_url=youtube_url,
                    provider=provider.lower(),
                    model=model,
                    api_key=api_key_to_use,
                    export_diagram=True,
                )

                if not result.get("is_informative", True):
                    st.session_state.rejected = True
                    st.session_state.rejection_msg = result.get("rejection_message", "")
                else:
                    st.session_state.explanation_text = result["explanation_text"]
                    st.session_state.iteration_count = result.get("iteration_count", 0)
                    st.session_state.generated = True
                    loops = result.get("iteration_count", 0) + 1
                    st.success(
                        f"✅ Explanation generated! "
                        f"({loops} research loop{'s' if loops > 1 else ''} completed)"
                    )

            except Exception as exc:
                st.error(f"❌ Error: {exc}")

if st.session_state.get("rejected"):
    st.warning("🚫 This video was flagged by the content guardrail.")
    st.markdown(
        f'<div class="guardrail-msg">{st.session_state.rejection_msg}</div>',
        unsafe_allow_html=True,
    )

if st.session_state.generated:
    if os.path.exists("architecture_diagram.png"):
        with st.expander("🗺️ Pipeline Architecture Diagram (v2)", expanded=False):
            st.image("architecture_diagram.png", caption="LangGraph v2 Pipeline")

    loops = st.session_state.iteration_count + 1
    if loops > 1:
        st.info(f"🔄 Pipeline completed **{loops} research loops** (verifier triggered re-research).")

    st.subheader("🎓 Demystified Explanation")
    st.markdown(st.session_state.explanation_text, unsafe_allow_html=True)
    st.download_button(
        label="📥 Download Demystified Markdown",
        data=st.session_state.explanation_text,
        file_name="demystified_explanation.md",
        mime="text/markdown",
    )

    st.markdown("---")
    st.subheader("🌐 Fast Translation")
    st.write("Translate without re-running the full pipeline.")

    col1, col2 = st.columns([1, 2])
    with col1:
        target_lang = st.selectbox(
            "Target Language",
            [
                "Spanish", "French", "German", "Hindi", "Telugu", "Tamil",
                "Chinese", "Japanese", "Russian", "Portuguese",
            ],
        )
        translate_btn = st.button("🌐 Translate Now", use_container_width=True)

    with col2:
        if translate_btn:
            src_text = st.session_state.explanation_text
            if not src_text:
                st.warning("No text to translate. Run the pipeline first.")
            else:
                with st.spinner(f"Translating to {target_lang} …"):
                    try:
                        api_key_to_use = custom_api_key.strip() or None
                        translated = translate_text(
                            text=src_text,
                            target_language=target_lang,
                            provider=provider.lower(),
                            model=model,
                            api_key=api_key_to_use,
                        )
                        st.subheader(f"Translated Output ({target_lang})")
                        st.markdown(translated)
                        st.download_button(
                            label=f"📥 Download ({target_lang})",
                            data=translated,
                            file_name=f"translated_demystified_{target_lang.lower()}.md",
                            mime="text/markdown",
                        )
                    except Exception as exc:
                        st.error(f"Translation error: {exc}")
