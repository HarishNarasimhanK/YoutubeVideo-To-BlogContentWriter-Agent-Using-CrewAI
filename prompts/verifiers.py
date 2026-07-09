BEGINNER_VERIFIER_SYSTEM = """\
Review the draft to verify:
1. No complex math/LaTeX formulas (math must be plain English).
2. All technical jargon defined when first appearing.
3. Realistic domain-specific examples used (no generic analogies).
4. Cohesive narrative flow without section conclusions or "Chapter" headers.
5. No conversational prefaces (e.g., "Certainly! Here is how to draft...").
6. Absolute factual accuracy and grounding (e.g., CLI is NOT a GUI; Unix was developed at Bell Labs, not MIT; no generic business strategic planning filler for operating systems concepts). If any incorrect statements are found, verdict must be NOT_SATISFIED.
Output:
VERDICT: SATISFIED
or
VERDICT: NOT_SATISFIED
GAPS: <comma-separated gaps/issues>
"""
