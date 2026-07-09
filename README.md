# 🎥 DemystifyAI

An autonomous multi-agent engineering pipeline designed to convert educational YouTube videos into comprehensive, first-principles explanations.

---

## 💡 The Problem & How We Are Different

When using general LLMs like ChatGPT to explain complex, technical content, they default to using academic jargon and mathematical formulas. To get ChatGPT to explain concepts from first principles without relying on confusing jargon, users have to write massive, custom prompts every single time.

**This agent solves that.** It is built to automatically demystify complex material by translating jargon into simple, plain-English explanations using real-world scenarios. 

* **Active Scope**: Educational YouTube video URLs.
* **Future Roadmap**: Support for uploaded research paper PDFs, documentation terms, and textbooks.

---

## 🏗️ Multi-Agent Architecture

```text
[Start] ──> extract_transcript ──> guardrail_check
                                        │
                                        ├── [Non-Informative] ──> [END: Rejected]
                                        ▼ [Informative]
                                   map_concepts
                                        │
                                        ▼
                                  shuffle_concepts
                                        │
                                        ▼
                                   plan_outline
                                        │
             ┌──────────────────────────┴ <─────────────────────────┐
             ▼                                                      │
          research                                                  │
             │                                                      │
             ▼                                                      │
        write_draft                                                 │ [Not Satisfied & Iterations < 3]
             │                                                      │
             ▼                                                      │
        verify_draft ──> check_quality ─────────────────────────────┘
                              │
                              ▼ [Satisfied / Max Iterations]
                         save_outputs ──> [END]
```

---

## 🚀 Running the Streamlit App

To run the interactive Streamlit dashboard:

```bash
.venv/bin/streamlit run app.py
```
