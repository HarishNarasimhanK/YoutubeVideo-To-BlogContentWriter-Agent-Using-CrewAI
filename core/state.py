from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict

class PipelineState(TypedDict):
    youtube_url: str
    provider: str
    model: str
    api_key: Optional[str]

    is_informative: Optional[bool]
    rejection_message: Optional[str]

    transcript: str
    global_context: str
    chunks: List[Dict[str, Any]]

    mapped_concepts_report: str
    structured_response_map: str
    outline: List[Dict[str, Any]]
    current_chapter_idx: int

    research_report: str
    research_feedback: Optional[str]
    iteration_count: int
    next_action: str

    draft: str

    final: str
    satisfied: bool
    gaps: str

    node_logs: Annotated[list, operator.add]
