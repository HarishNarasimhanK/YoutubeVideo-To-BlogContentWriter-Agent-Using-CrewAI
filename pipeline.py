from __future__ import annotations

from typing import Optional

from graph.builder import build_graph, export_architecture_diagram
from core.state import PipelineState
from graph.nodes import run_translate

_graph = None

def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph

def run_pipeline(
    source_input: str,
    source_type: str,
    provider: str,
    model: str,
    api_key: Optional[str] = None,
    export_diagram: bool = True,
) -> dict:
    """Run the full DemystifyAI LangGraph pipeline."""
    graph = _get_graph()

    if export_diagram:
        export_architecture_diagram(graph)

    initial_state: PipelineState = {
        "source_input":             source_input,
        "source_type":              source_type,
        "source_label":             "",
        "provider":                 provider,
        "model":                    model,
        "api_key":                  api_key,
        "is_informative":           None,
        "rejection_message":        None,
        "transcript":               "",
        "global_context":           "",
        "chunks":                   [],
        "mapped_concepts_report":   "",
        "structured_response_map":  "",
        "outline":                  [],
        "current_chapter_idx":      0,
        "research_report":          "",
        "research_feedback":        None,
        "iteration_count":          0,
        "next_action":              "save_outputs",
        "draft":                    "",
        "final":                    "",
        "satisfied":                False,
        "gaps":                     "",
        "node_logs":                [],
    }

    final_state = graph.invoke(initial_state)

    return {
        "is_informative":    final_state.get("is_informative", True),
        "rejection_message": final_state.get("rejection_message"),
        "explanation_file":  "demystified_explanation.md",
        "explanation_text":  final_state.get("final", ""),
        "node_logs":         final_state.get("node_logs",        []),
        "iteration_count":   final_state.get("iteration_count",  0),
    }

def translate_text(
    text: str,
    target_language: str,
    provider: str,
    model: str,
    api_key: Optional[str] = None,
) -> str:
    """Translate a Markdown document without re-running the full pipeline."""
    return run_translate(
        text=text,
        language=target_language,
        provider=provider,
        model=model,
        api_key=api_key,
    )
