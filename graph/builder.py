from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from core.state import PipelineState
from graph.nodes import (
    check_quality_node,
    extract_transcript_node,
    guardrail_node,
    map_concepts_node,
    shuffle_concepts_node,
    plan_outline_node,
    research_node,
    save_outputs_node,
    verify_draft_node,
    write_draft_node,
)

def _route_after_guardrail(state: PipelineState) -> str:
    """Route execution after guardrail classification."""
    if state.get("is_informative", True):
        return "map_concepts"
    return END

def _route_after_quality_check(state: PipelineState) -> str:
    """Route execution after verifier quality check."""
    return state.get("next_action", "save_outputs")

def build_graph() -> StateGraph:
    """Build and compile the execution graph."""
    builder = StateGraph(PipelineState)

    builder.add_node("extract_transcript", extract_transcript_node)
    builder.add_node("guardrail",          guardrail_node)
    builder.add_node("map_concepts",       map_concepts_node)
    builder.add_node("shuffle_concepts",   shuffle_concepts_node)
    builder.add_node("plan_outline",       plan_outline_node)
    builder.add_node("research",           research_node)
    builder.add_node("write_draft",        write_draft_node)
    builder.add_node("verify_draft",       verify_draft_node)
    builder.add_node("check_quality",      check_quality_node)
    builder.add_node("save_outputs",       save_outputs_node)

    builder.add_edge(START,                "extract_transcript")
    builder.add_edge("extract_transcript", "guardrail")

    builder.add_conditional_edges(
        "guardrail",
        _route_after_guardrail,
        {"map_concepts": "map_concepts", END: END},
    )

    builder.add_edge("map_concepts",       "shuffle_concepts")
    builder.add_edge("shuffle_concepts",   "plan_outline")
    builder.add_edge("plan_outline",       "research")

    builder.add_edge("research",           "write_draft")
    builder.add_edge("write_draft",        "verify_draft")

    builder.add_edge("verify_draft",       "check_quality")

    builder.add_conditional_edges(
        "check_quality",
        _route_after_quality_check,
        {"research": "research", "save_outputs": "save_outputs"},
    )

    builder.add_edge("save_outputs", END)

    return builder.compile()

def export_architecture_diagram(graph, output_png: str = "architecture_diagram.png") -> str:
    """Export diagram of the graph as PNG and Mermaid format."""
    try:
        png_bytes = graph.get_graph().draw_mermaid_png()
        with open(output_png, "wb") as fh:
            fh.write(png_bytes)

        mmd_path = output_png.replace(".png", ".mmd")
        mermaid_src = graph.get_graph().draw_mermaid()
        with open(mmd_path, "w", encoding="utf-8") as fh:
            fh.write(mermaid_src)

        return output_png
    except Exception:
        ascii_art = graph.get_graph().draw_ascii()
        return ascii_art
