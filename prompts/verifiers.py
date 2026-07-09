BEGINNER_VERIFIER_SYSTEM = """\
Review the draft and verify:
1. No complex math or LaTeX formulas (all math explained in plain English).
2. Every technical term is defined when it first appears.
3. Realistic, domain-specific examples used (no generic analogies).
4. Cohesive narrative flow without section-level conclusions or "Chapter" headers.
5. No conversational prefaces (e.g., "Certainly!", "Here is how to draft...").
6. The draft must contain ONLY facts directly supported by the source text or research report. Reject any content containing ungrounded claims, generic textbook filler, or topics not present in the source material.
Output:
VERDICT: SATISFIED
or
VERDICT: NOT_SATISFIED
GAPS: <comma-separated gaps/issues>
"""
