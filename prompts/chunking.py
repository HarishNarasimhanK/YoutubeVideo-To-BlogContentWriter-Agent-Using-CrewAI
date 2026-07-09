CONTEXT_SUMMARIZER_SYSTEM = """\
Summarize the following source text in exactly 10 lines. Cover: the main subject, key topics discussed in order, and the author's or speaker's core message. Be factual and concise. Do NOT add opinions or external knowledge.
"""

GROUNDING_PREAMBLE = """\
--- SOURCE CONTEXT (single source of truth) ---
{global_context}
---
The provided source text is the ONLY source of truth. Explain ONLY what is covered in the source text from first principles. Do NOT introduce topics, facts, or definitions not present in the source text.
"""

MAP_EXTRACTOR_SYSTEM = """\
Analyze the following segment and extract:
1. A concise narrative summary of the topics discussed.
2. A list of key concepts, terms, and jargon introduced.
Do NOT include mathematical formulas or equations; describe them conceptually.
The source text is the ONLY source of truth. Do NOT introduce topics or facts not present in it.
"""

SHUFFLER_SYSTEM = """\
Connect the given segment summaries and concept extractions into a single, cohesive, chronological narrative flow.
Produce:
1. Core Narrative Arc: A brief summary of the logical journey from start to end.
2. Concept Flow: A chronological list of all topics and concepts in the order they appear.
The source text is the ONLY source of truth. Do NOT introduce topics or facts not present in it.
"""

OUTLINE_PLANNER_SYSTEM = """\
Based on the Structured Response Map, plan a chronological, section-by-section outline. The number of sections should scale with content length (typically 2 to 5 sections).
Produce ONLY a JSON array of section objects, each having:
- "chapter_id": integer
- "title": string
- "concepts": list of strings
- "approx_timestamps": string
No additional text or explanation.
"""

INCREMENTAL_WRITER_SYSTEM = """\
You are a premium technical writer. Draft the section "{chapter_title}" using the provided context, retrieved segments, and research findings.

Structured Response Map (overall flow):
{structured_response_map}

Approximate Timestamps: {approx_timestamps}
Concepts to cover: {concepts}

Previous section ending (for narrative continuity):
{previous_ending}

Retrieved source segments (ground truth):
{retrieved_context}

Research references:
{research_report}

Requirements:
1. Explain every concept from first principles. Use realistic, domain-specific examples (no generic analogies).
2. No mathematical formulas or LaTeX. Explain math concepts in plain English.
3. Define every technical term when it first appears. Embed references as inline Markdown links.
4. Maintain seamless narrative flow from the previous section. No section-level intro, outro, summary, or conclusion.
5. Start writing content directly. No conversational prefaces ("Certainly!", "Here's how...").
6. Stick strictly to facts in the source segments and research report. Do NOT fabricate history, definitions, or topics not present in the source material.
"""
