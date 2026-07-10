from __future__ import annotations

import concurrent.futures
import json
import re
import time
from typing import Any, Union

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import LLMResult
from langchain_ollama import ChatOllama

from core.state import PipelineState
from core.llm import get_llm, get_max_tokens_per_chunk
from prompts import (
    BEGINNER_VERIFIER_SYSTEM,
    BEGINNER_WRITER_SYSTEM,
    CONTEXT_SUMMARIZER_SYSTEM,
    GROUNDING_PREAMBLE,
    GUARDRAIL_REJECTION_TEMPLATE,
    GUARDRAIL_SYSTEM,
    RESEARCHER_SYSTEM,
    RESEARCHER_WITH_FEEDBACK_SYSTEM,
    TRANSLATOR_SYSTEM,
    MAP_EXTRACTOR_SYSTEM,
    SHUFFLER_SYSTEM,
    OUTLINE_PLANNER_SYSTEM,
    INCREMENTAL_WRITER_SYSTEM,
)
from core.extractors import get_extractor
from tools.search import search_arxiv, search_wikipedia
from tools.rag import TranscriptRAGStore

MAX_LOOP_ITERATIONS = 3

class VerboseCallbackHandler(BaseCallbackHandler):
    """Prints LLM prompt, response, and errors to stdout."""
    _W = 72

    def __init__(self, node_name: str = "") -> None:
        super().__init__()
        self.node_name = node_name

    def on_chat_model_start(self, serialized: dict, messages: list, **kwargs: Any) -> None:
        model_id = (
            serialized.get("kwargs", {}).get("model_name")
            or serialized.get("kwargs", {}).get("model")
            or serialized.get("name", "LLM")
        )
        print(f"\n{'─' * self._W}")
        print(f"  🤖  CHAT MODEL → [{self.node_name}]  |  {model_id}")
        print(f"{'─' * self._W}")
        for msg_group in messages:
            for msg in msg_group:
                role = type(msg).__name__.replace("Message", "").upper()
                content = str(msg.content)
                preview = content[:2500] + ("  …[truncated]" if len(content) > 2500 else "")
                print(f"\n📤 {role}:\n{preview}")
        print()

    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs: Any) -> None:
        model_id = serialized.get("name", "LLM")
        print(f"\n{'─' * self._W}")
        print(f"  🤖  LLM → [{self.node_name}]  |  {model_id}")
        print(f"{'─' * self._W}")
        for i, prompt in enumerate(prompts):
            preview = prompt[:2500] + ("  …[truncated]" if len(prompt) > 2500 else "")
            print(f"\n📤 PROMPT [{i + 1}]:\n{preview}")
        print()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        print(f"\n{'─' * self._W}")
        print(f"  ✅  LLM RESPONSE ← [{self.node_name}]")
        print(f"{'─' * self._W}")
        for gen_group in response.generations:
            for gen in gen_group:
                text = getattr(gen, "text", None)
                if not text and hasattr(gen, "message"):
                    text = getattr(gen.message, "content", "") or ""
                if not text:
                    text = str(gen)
                preview = text[:4000] + ("  …[truncated]" if len(text) > 4000 else "")
                print(f"\n📥 RESPONSE:\n{preview}\n")

    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        print(f"\n❌  LLM ERROR [{self.node_name}]: {error}")

def _banner(node_name: str, status: str = "START", elapsed: float | None = None) -> None:
    W = 72
    if status == "START":
        print(f"\n{'═' * W}")
        print(f"  🚀  NODE STARTED  :  {node_name}")
        print(f"{'═' * W}")
    else:
        suffix = f"  |  ⏱ {elapsed:.1f}s" if elapsed is not None else ""
        print(f"\n{'═' * W}")
        print(f"  ✅  NODE COMPLETE :  {node_name}{suffix}")
        print(f"{'═' * W}")

def _invoke_with_retry(
    llm: Any,
    messages: list,
    config: dict | None = None,
    max_retries: int = 8,
    initial_backoff: float = 15.0,
) -> Any:
    """Invoke an LLM with exponential back-off retry."""
    if isinstance(llm, ChatOllama):
        time.sleep(2.0)

    backoff = initial_backoff
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages, config=config or {})
        except Exception as e:
            err = str(e).lower()
            is_rate_limit = any(k in err for k in ["rate_limit", "rate limit", "429", "tpm", "ratelimit", "limit reached", "resource_exhausted"])
            is_network_err = any(k in err for k in ["disconnected", "connection", "remote protocol", "timeout", "refused", "reset by peer", "httpx", "httpcore"])
            
            if is_rate_limit or is_network_err:
                if attempt == max_retries - 1:
                    raise
                
                sleep_time = backoff
                if is_rate_limit:
                    if "ms" in err:
                        sleep_time = 2.0
                    else:
                        match = re.search(r"(?:retry in|try again in) (?:(\d+)h)?(?:(\d+)m)?(?:([\d\.]+)s)?", err)
                        if match:
                            h, m, s = match.groups()
                            parsed_secs = 0.0
                            if h:
                                parsed_secs += float(h) * 3600.0
                            if m:
                                parsed_secs += float(m) * 60.0
                            if s:
                                parsed_secs += float(s)
                            if parsed_secs > 0:
                                sleep_time = parsed_secs + 5.0

                    print(f"\n⚠️  Rate limit hit. Attempt {attempt + 1}/{max_retries}. Sleeping {sleep_time:.1f}s...")
                else:
                    print(f"\n⚠️  Connection error: {e}. Attempt {attempt + 1}/{max_retries}. Sleeping {sleep_time:.1f}s...")
                
                time.sleep(sleep_time)
                backoff = min(backoff * 1.5, 120.0)
            else:
                raise

def _parse_guardrail_response(text: Any) -> tuple[bool, str]:
    """Parse JSON guardrail classification response."""
    if isinstance(text, list):
        parts = []
        for part in text:
            if isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
        text = "".join(parts)
    elif not isinstance(text, str):
        text = str(text)

    cleaned = re.sub(r"```[a-z]*\n?", "", text).strip()
    match = re.search(r"\{[^{}]+\}", cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            verdict = str(data.get("verdict", "INFORMATIVE")).upper().strip()
            reason = str(data.get("reason", ""))
            return verdict == "INFORMATIVE", reason
        except (json.JSONDecodeError, KeyError):
            pass

    upper = text.upper()
    if "NON_INFORMATIVE" in upper or "NON-INFORMATIVE" in upper:
        return False, "Classified as non-informative."

    print(f"[guardrail] ⚠️  Could not parse JSON response — defaulting to INFORMATIVE.")
    return True, "Unable to parse guardrail response — defaulting to allow."

def _parse_verifier_response(response_text: str, verifier_name: str = "verifier") -> tuple[bool, str, str]:
    """Parse structured verifier verdict."""
    text = response_text.strip()
    satisfied = False
    gaps = ""
    found_verdict = False
    
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in lines:
        if ":" in line:
            key, _, val = line.partition(":")
            key = key.strip().upper()
            val = val.strip()
            if "VERDICT" in key:
                found_verdict = True
                satisfied = ("SATISFIED" in val.upper() and "NOT" not in val.upper())
            elif "GAPS" in key:
                gaps = val
                
    if not found_verdict:
        upper_text = text.upper()
        if "NOT_SATISFIED" in upper_text or "NOT SATISFIED" in upper_text:
            satisfied = False
        elif "SATISFIED" in upper_text:
            satisfied = True
            
        gaps_match = re.search(r"(?:GAPS|Gaps|gaps)\s*:\s*(.*)", text)
        if gaps_match:
            gaps = gaps_match.group(1).strip()
            satisfied = False

    if satisfied:
        gaps = ""

    verdict_label = "SATISFIED" if satisfied else f"NOT_SATISFIED (gaps: {gaps})"
    print(f"[{verifier_name}] Verdict → {verdict_label}")

    return satisfied, gaps, ""

def extract_text_node(state: PipelineState) -> dict:
    """Extract text from the input source using the appropriate extractor."""
    t0 = time.time()
    _banner("extract_text_node")

    source_type = state["source_type"]
    source_input = state["source_input"]
    print(f"[extract_text_node] Source type: {source_type}")

    max_tokens = get_max_tokens_per_chunk(state["provider"], state["model"])
    print(f"[extract_text_node] Calculated max chunk tokens: {max_tokens}")

    extractor = get_extractor(source_type)
    content = extractor.extract(source_input, max_tokens_per_chunk=max_tokens)

    transcript = content.text
    chunks = content.chunks
    source_label = content.source_label

    print(f"[extract_text_node] {source_label}: {len(transcript):,} chars, {len(chunks)} chunks.")

    print(f"[extract_text_node] Generating global context summary …")
    llm = get_llm(state["provider"], state["model"], state["api_key"])
    handler = VerboseCallbackHandler("context_summarizer")
    context_excerpt = transcript[:6000]
    context_messages = [
        SystemMessage(content=CONTEXT_SUMMARIZER_SYSTEM),
        HumanMessage(content=f"Source text:\n{context_excerpt}"),
    ]
    context_response = _invoke_with_retry(llm, context_messages, config={"callbacks": [handler]})
    global_context = context_response.content.strip()
    print(f"[extract_text_node] Global context ({len(global_context)} chars) generated.")

    elapsed = time.time() - t0
    _banner("extract_text_node", "COMPLETE", elapsed)

    return {
        "transcript": transcript,
        "global_context": global_context,
        "source_label": source_label,
        "chunks": chunks,
        "node_logs": [f"[extract_text_node] {source_label} | {len(transcript):,} chars in {elapsed:.1f}s"],
    }

def guardrail_node(state: PipelineState) -> dict:
    """Determine if transcript is informative or non-informative."""
    t0 = time.time()
    _banner("guardrail_node")

    excerpt = state["transcript"][:3000]
    print(f"[guardrail_node] Classifying {len(excerpt):,}-char transcript excerpt …")

    llm = get_llm(state["provider"], state["model"], state["api_key"])
    handler = VerboseCallbackHandler("guardrail_node")

    messages = [
        SystemMessage(content=GUARDRAIL_SYSTEM),
        HumanMessage(content=(
            "Classify this YouTube video transcript excerpt:\n\n"
            f"--- TRANSCRIPT START ---\n{excerpt}\n--- TRANSCRIPT END ---"
        )),
    ]

    response = _invoke_with_retry(llm, messages, config={"callbacks": [handler]})
    is_informative, reason = _parse_guardrail_response(response.content)
    elapsed = time.time() - t0

    if is_informative:
        print(f"[guardrail_node] ✅ INFORMATIVE — {reason}")
        _banner("guardrail_node", "COMPLETE", elapsed)
        return {
            "is_informative": True,
            "rejection_message": None,
            "node_logs": [f"[guardrail_node] INFORMATIVE ({reason}) | {elapsed:.1f}s"],
        }
    else:
        rejection = GUARDRAIL_REJECTION_TEMPLATE.format(reason=reason)
        print(f"[guardrail_node] 🚫 NON_INFORMATIVE — {reason}")
        _banner("guardrail_node", "COMPLETE", elapsed)
        return {
            "is_informative": False,
            "rejection_message": rejection,
            "node_logs": [f"[guardrail_node] NON_INFORMATIVE ({reason}) | {elapsed:.1f}s — pipeline aborted"],
        }

def map_concepts_node(state: PipelineState) -> dict:
    """Map over transcript chunks to extract concepts."""
    t0 = time.time()
    _banner("map_concepts_node")

    llm = get_llm(state["provider"], state["model"], state["api_key"])
    handler = VerboseCallbackHandler("map_concepts_node")

    chunks = state["chunks"]
    global_context = state.get("global_context", "")
    grounding = GROUNDING_PREAMBLE.format(global_context=global_context) if global_context else ""
    print(f"[map_concepts_node] Mapping over {len(chunks)} transcript chunks …")

    def process_chunk(idx, chunk):
        system_prompt = grounding + "\n" + MAP_EXTRACTOR_SYSTEM if grounding else MAP_EXTRACTOR_SYSTEM
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=(
                f"Segment Index: {idx + 1}\n"
                f"Timestamps: {chunk['start_time']:.1f}s - {chunk['end_time']:.1f}s\n\n"
                f"Transcript excerpt:\n{chunk['text']}"
            ))
        ]
        response = _invoke_with_retry(llm, messages, config={"callbacks": [handler]})
        return f"### Segment {idx + 1} ({chunk['start_time']:.1f}s - {chunk['end_time']:.1f}s)\n{response.content}"

    max_workers = 1 if state["provider"] == "ollama" else 5
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_chunk, idx, chunk) for idx, chunk in enumerate(chunks)]
        mapped_summaries = [f.result() for f in futures]

    mapped_text = "\n\n---\n\n".join(mapped_summaries)
    elapsed = time.time() - t0
    _banner("map_concepts_node", "COMPLETE", elapsed)

    return {
        "mapped_concepts_report": mapped_text,
        "node_logs": [f"[map_concepts_node] Mapped {len(chunks)} chunks | {elapsed:.1f}s"],
    }

def shuffle_concepts_node(state: PipelineState) -> dict:
    """Connect mapped concepts into a cohesive Structured Response Map."""
    t0 = time.time()
    _banner("shuffle_concepts_node")

    llm = get_llm(state["provider"], state["model"], state["api_key"])
    handler = VerboseCallbackHandler("shuffle_concepts_node")

    mapped_text = state.get("mapped_concepts_report") or ""
    messages = [
        SystemMessage(content=SHUFFLER_SYSTEM),
        HumanMessage(content=(
            "Here are the segment summaries and extracted concepts. Align and connect them:\n\n"
            f"{mapped_text}"
        ))
    ]
    response = _invoke_with_retry(llm, messages, config={"callbacks": [handler]})
    elapsed = time.time() - t0
    _banner("shuffle_concepts_node", "COMPLETE", elapsed)

    return {
        "structured_response_map": response.content,
        "node_logs": [f"[shuffle_concepts_node] Compiled Structured Response Map | {elapsed:.1f}s"]
    }

def plan_outline_node(state: PipelineState) -> dict:
    """Plan section outline dynamically based on Structured Response Map."""
    t0 = time.time()
    _banner("plan_outline_node")

    llm = get_llm(state["provider"], state["model"], state["api_key"])
    handler = VerboseCallbackHandler("plan_outline_node")

    response_map = state["structured_response_map"]
    messages = [
        SystemMessage(content=OUTLINE_PLANNER_SYSTEM),
        HumanMessage(content=(
            "Create outline from Structured Response Map as a JSON array of objects:\n\n"
            f"{response_map}"
        ))
    ]
    response = _invoke_with_retry(llm, messages, config={"callbacks": [handler]})

    text = response.content
    cleaned = re.sub(r"```json\n?", "", text).replace("```", "").strip()
    try:
        outline = json.loads(cleaned)
        if not isinstance(outline, list):
            raise ValueError("Outline must be a list of sections")
    except Exception:
        outline = [
            {
                "chapter_id": 1,
                "title": "Definitions & Core Principles",
                "concepts": ["Definitions"],
                "approx_timestamps": "Full Video"
            },
            {
                "chapter_id": 2,
                "title": "Detailed Video Breakdown",
                "concepts": ["Detailed Explanation"],
                "approx_timestamps": "Full Video"
            }
        ]

    elapsed = time.time() - t0
    _banner("plan_outline_node", "COMPLETE", elapsed)

    return {
        "outline": outline,
        "current_chapter_idx": 0,
        "node_logs": [f"[plan_outline_node] Planned {len(outline)} chapters | {elapsed:.1f}s"]
    }

def research_node(state: PipelineState) -> dict:
    """Identify and research concepts using external references."""
    t0 = time.time()
    iteration = state.get("iteration_count", 0)
    feedback = state.get("research_feedback") or ""
    is_retry = iteration > 0 and bool(feedback)

    _banner(f"research_node  [iteration {iteration + 1}]")

    model_lower = state["model"].lower()
    is_incapable_ollama = (
        state["provider"] == "ollama" 
        and ("llama3:" in model_lower or "llama3.0" in model_lower or "llama-3-8b" in model_lower)
    )
    use_native_tools = False if state["provider"] == "ollama" else not is_incapable_ollama

    llm = get_llm(state["provider"], state["model"], state["api_key"])
    tools = [search_wikipedia, search_arxiv]
    tool_map = {t.name: t for t in tools}

    if use_native_tools:
        llm_invocable = llm.bind_tools(tools)
    else:
        llm_invocable = llm

    store = TranscriptRAGStore(state["chunks"], state["provider"], state["api_key"])
    query = feedback if is_retry else (state.get("structured_response_map", "")[:500] or "core concepts")
    retrieved = store.retrieve(query, top_k=5)
    
    excerpt = "\n\n".join(
        f"[Timestamps: {c['start_time']:.1f}s - {c['end_time']:.1f}s]\n{c['text']}"
        for c in retrieved
    )

    global_context = state.get("global_context", "")
    grounding = GROUNDING_PREAMBLE.format(global_context=global_context) if global_context else ""

    system_prompt = RESEARCHER_WITH_FEEDBACK_SYSTEM if is_retry else RESEARCHER_SYSTEM
    system_prompt = grounding + "\n" + system_prompt if grounding else system_prompt
    if not use_native_tools:
        system_prompt += """

── TOOL CALLING INSTRUCTIONS ──────────────────────────────────────────
You have access to two tools to search for verified academic references:
1. `search_wikipedia` (parameters: `query` string)
2. `search_arxiv` (parameters: `query` string)

To call a tool, you MUST output a line in exactly this format:
TOOL_CALL: <tool_name>("<query>")
For example:
TOOL_CALL: search_wikipedia("support vector machine")
TOOL_CALL: search_arxiv("training optimal margin classifiers Vapnik")

You can make multiple tool calls in a single response.
If you have retrieved all the necessary information and are ready to write the final reference guide, do NOT output any TOOL_CALL lines. Simply write the final markdown reference guide.
"""

    human_content = (
        "Identify core technical concepts and research them using your tools:\n\n"
        f"--- TRANSCRIPT SEGMENTS START ---\n{excerpt}\n--- TRANSCRIPT SEGMENTS END ---"
    )
    if is_retry and feedback:
        human_content += (
            "\n\n--- VERIFIER FEEDBACK (RESEARCH GAPS) ---\n"
            f"{feedback}\n"
            "--- END FEEDBACK ---\n\n"
            "⚠️  You MUST specifically search for each topic listed above as a priority."
        )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_content),
    ]

    handler = VerboseCallbackHandler(f"research_node[iter={iteration + 1}]")
    config = {"callbacks": [handler]}

    MAX_ITER = 10
    response = None

    for i in range(MAX_ITER):
        print(f"\n[research_node] ── ReAct sub-iteration {i + 1}/{MAX_ITER} ──")
        response = _invoke_with_retry(llm_invocable, messages, config=config)
        messages.append(response)

        tool_calls = []
        if use_native_tools:
            if response.tool_calls:
                for tc in response.tool_calls:
                    tool_calls.append({
                        "id": tc["id"],
                        "name": tc["name"],
                        "args": tc["args"]
                    })
        else:
            matches = list(re.finditer(r"TOOL_CALL:\s*(\w+)\((.*?)\)", response.content or ""))
            for match in matches:
                t_name = match.group(1).strip()
                t_args_str = match.group(2).strip().strip("'\"")
                tool_calls.append({
                    "id": f"manual_{time.time()}_{t_name}",
                    "name": t_name,
                    "args": {"query": t_args_str}
                })

        if not tool_calls:
            break

        tool_outputs = []
        for tc in tool_calls:
            t_name = tc["name"]
            t_args = tc["args"]
            t_id = tc["id"]

            tool_fn = tool_map.get(t_name)
            if not tool_fn:
                if "wikipedia" in t_name.lower():
                    tool_fn = search_wikipedia
                elif "arxiv" in t_name.lower():
                    tool_fn = search_arxiv

            result_str = ""
            if tool_fn:
                try:
                    result = tool_fn.invoke(t_args)
                    result_str = str(result)
                except Exception as ex:
                    result_str = f"Error executing tool: {ex}"
            else:
                result_str = f"Unknown tool '{t_name}'."

            if use_native_tools:
                messages.append(ToolMessage(content=result_str, tool_call_id=t_id))
            else:
                tool_outputs.append(f"Tool Output for {t_name}({t_args}):\n{result_str}")

        if not use_native_tools and tool_outputs:
            messages.append(HumanMessage(content="\n\n".join(tool_outputs)))

    research_report = (response.content or "") if response else "Research failed."
    elapsed = time.time() - t0
    _banner(f"research_node  [iteration {iteration + 1}]", "COMPLETE", elapsed)

    return {
        "research_report": research_report,
        "node_logs": [f"[research_node] iter={iteration + 1} | {i + 1} ReAct sub-iters | {len(research_report):,} chars | {elapsed:.1f}s"],
    }

def prune_repetitions(text: str) -> str:
    """Prune LLM repetition patterns paragraph by paragraph."""
    paragraphs = text.split("\n\n")
    paragraph_counts = {}
    cleaned_paragraphs = []
    for p in paragraphs:
        p_clean = p.strip().lower()
        if not p_clean:
            continue
        if len(p_clean) < 20:
            cleaned_paragraphs.append(p)
            continue
        paragraph_counts[p_clean] = paragraph_counts.get(p_clean, 0) + 1
        if paragraph_counts[p_clean] >= 3 or (paragraph_counts[p_clean] >= 2 and len(p_clean) > 150):
            break
        cleaned_paragraphs.append(p)
    return "\n\n".join(cleaned_paragraphs)

def write_draft_node(state: PipelineState) -> dict:
    """Draft technical explanation sections using outline & retrieved RAG context."""
    t0 = time.time()
    iteration = state.get("iteration_count", 0)
    _banner(f"write_draft_node  [iteration {iteration + 1}]")

    llm = get_llm(state["provider"], state["model"], state["api_key"])
    handler = VerboseCallbackHandler(f"write_draft_node[iter={iteration + 1}]")
    store = TranscriptRAGStore(state["chunks"], state["provider"], state["api_key"])

    global_context = state.get("global_context", "")
    grounding = GROUNDING_PREAMBLE.format(global_context=global_context) if global_context else ""

    outline = state.get("outline") or []
    if not outline:
        messages = [
            SystemMessage(content=BEGINNER_WRITER_SYSTEM),
            HumanMessage(content=(
                f"TRANSCRIPT:\n{state['transcript'][:5000]}\n\n"
                f"RESEARCH REPORT:\n{state['research_report']}"
            )),
        ]
        response = _invoke_with_retry(llm, messages, config={"callbacks": [handler]})
        content = response.content
    else:
        compiled_sections = []
        previous_ending = "None."

        for idx, sec in enumerate(outline):
            title = sec.get("title", f"Section {idx + 1}")
            approx_ts = sec.get("approx_timestamps", "N/A")
            concepts = ", ".join(sec.get("concepts", []))

            retrieved = store.retrieve(f"{title} {concepts}", top_k=3)
            retrieved_context = "\n\n".join(
                f"[Segment Timestamps: {c['start_time']:.1f}s - {c['end_time']:.1f}s]\n{c['text']}"
                for c in retrieved
            )

            base_prompt = INCREMENTAL_WRITER_SYSTEM.format(
                structured_response_map=state.get("structured_response_map", "N/A"),
                chapter_title=title,
                approx_timestamps=approx_ts,
                concepts=concepts,
                previous_ending=previous_ending,
                retrieved_context=retrieved_context,
                research_report=state["research_report"]
            )
            system_prompt = grounding + "\n" + base_prompt if grounding else base_prompt

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Draft the content for the section: {title}")
            ]

            response = _invoke_with_retry(llm, messages, config={"callbacks": [handler]})
            section_content = response.content.strip()
            section_content = prune_repetitions(section_content)

            clean_lines = []
            skip = False
            for line in section_content.splitlines():
                line_stripped = line.strip().lower().replace("*", "").replace("#", "").strip()
                if line_stripped in ["conclusion", "references", "summary", "takeaways", "final thoughts"]:
                    skip = True
                    continue
                if skip:
                    if line.strip().startswith("#"):
                        skip = False
                    else:
                        continue
                clean_lines.append(line)
            section_content = "\n".join(clean_lines).strip()

            sec_header = f"## {title}"
            lines = [line for line in section_content.splitlines()]
            if lines:
                first_line = lines[0].strip()
                is_bold_title = first_line.startswith("**") and first_line.endswith("**") and len(first_line) < 100
                is_md_header = first_line.startswith("#") and len(first_line) < 100
                
                if is_bold_title or is_md_header:
                    first_line_clean = first_line.strip("#*_ ").lower()
                    title_clean = title.strip().lower()
                    title_words = set(title_clean.split())
                    first_words = set(first_line_clean.split())
                    common_words = title_words.intersection(first_words)
                    
                    if len(common_words) >= min(2, len(title_words)) or title_clean in first_line_clean or first_line_clean in title_clean:
                        section_content = "\n".join(lines[1:]).strip()
            
            section_content = f"{sec_header}\n{section_content}"
            compiled_sections.append(section_content)

            lines = section_content.splitlines()
            last_lines = [l for l in lines[-4:] if l.strip()]
            previous_ending = "\n".join(last_lines) if last_lines else section_content[-300:]

        content = "\n\n".join(compiled_sections)

    elapsed = time.time() - t0
    _banner(f"write_draft_node  [iteration {iteration + 1}]", "COMPLETE", elapsed)

    return {
        "draft": content,
        "node_logs": [f"[write_draft_node] iter={iteration + 1} | {len(content):,} chars | {elapsed:.1f}s"],
    }

def verify_draft_node(state: PipelineState) -> dict:
    """Verify and check draft quality constraints."""
    t0 = time.time()
    iteration = state.get("iteration_count", 0)
    _banner(f"verify_draft_node  [iteration {iteration + 1}]")

    llm = get_llm(state["provider"], state["model"], state["api_key"])
    handler = VerboseCallbackHandler(f"verify_draft_node[iter={iteration + 1}]")

    global_context = state.get("global_context", "")
    grounding = GROUNDING_PREAMBLE.format(global_context=global_context) if global_context else ""
    verifier_prompt = grounding + "\n" + BEGINNER_VERIFIER_SYSTEM if grounding else BEGINNER_VERIFIER_SYSTEM

    messages = [
        SystemMessage(content=verifier_prompt),
        HumanMessage(content="Review draft:\n\n" + state["draft"]),
    ]

    response = _invoke_with_retry(llm, messages, config={"callbacks": [handler]})
    satisfied, gaps_found, _ = _parse_verifier_response(response.content, "verify_draft_node")
    elapsed = time.time() - t0
    _banner(f"verify_draft_node  [iteration {iteration + 1}]", "COMPLETE", elapsed)

    return {
        "final": state["draft"],
        "satisfied": satisfied,
        "gaps": gaps_found,
        "node_logs": [f"[verify_draft_node] iter={iteration + 1} | {'SATISFIED' if satisfied else f'NOT_SATISFIED gaps=[{gaps_found}]'} | {len(state['draft']):,} chars | {elapsed:.1f}s"],
    }

def check_quality_node(state: PipelineState) -> dict:
    """Assess verification output and route next pipeline execution step."""
    _banner("check_quality_node")
    ok = state.get("satisfied", False)
    iteration = state.get("iteration_count", 0)

    if not ok and iteration < MAX_LOOP_ITERATIONS:
        gaps_found = state.get("gaps", "").strip()
        feedback = f"[Verifier gaps]: {gaps_found}" if gaps_found else "Research the core concepts more thoroughly."
        new_iter = iteration + 1
        return {
            "next_action": "research",
            "iteration_count": new_iter,
            "research_feedback": feedback,
            "node_logs": [f"[check_quality_node] iter={new_iter} LOOP BACK | feedback: {feedback[:120]}"],
        }

    if ok:
        reason = f"verifier SATISFIED on iteration {iteration + 1}"
    else:
        reason = f"max iterations ({MAX_LOOP_ITERATIONS}) reached"

    return {
        "next_action": "save_outputs",
        "node_logs": [f"[check_quality_node] → save_outputs | {reason}"],
    }

def save_outputs_node(state: PipelineState) -> dict:
    """Save finalized markdown guide to files."""
    t0 = time.time()
    _banner("save_outputs_node")

    content = state.get("final", "")
    fname = "demystified_explanation.md"

    with open(fname, "w", encoding="utf-8") as f:
        f.write(content)

    elapsed = time.time() - t0
    total_loops = state.get("iteration_count", 0) + 1
    _banner("save_outputs_node", "COMPLETE", elapsed)

    return {
        "node_logs": [f"[save_outputs_node] Files saved in {elapsed:.1f}s | {total_loops} loop(s)"],
    }

def run_translate(text: str, language: str, provider: str, model: str, api_key: str | None = None) -> str:
    """Translate markdown text to target language."""
    t0 = time.time()
    _banner(f"translate → {language}")

    llm = get_llm(provider, model, api_key)
    handler = VerboseCallbackHandler(f"translate[{language}]")

    messages = [
        SystemMessage(content=TRANSLATOR_SYSTEM.format(language=language)),
        HumanMessage(content=text),
    ]

    response = _invoke_with_retry(llm, messages, config={"callbacks": [handler]})
    elapsed = time.time() - t0
    _banner(f"translate → {language}", "COMPLETE", elapsed)

    return response.content
