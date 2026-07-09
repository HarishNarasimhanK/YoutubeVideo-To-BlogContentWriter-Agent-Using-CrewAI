#!/usr/bin/env python3

import argparse
import json
import os
import socket
import sys
import time
import urllib.request
from typing import Any

socket.setdefaulttimeout(120)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.state import PipelineState
from graph import nodes

MOCK_TRANSCRIPT = """
PATRICK WINSTON: Today we are going to discuss Support Vector Machines (SVMs).
SVMs are powerful supervised learning models used for classification and regression.
The key idea is to find a decision boundary (a hyperplane) that separates classes.
Specifically, we want to find the separator that maximizes the distance between the closest points of each class.
This distance is called the margin, and the closest points are the support vectors.
By maximizing the margin, we create a 'widest street' separating our positive and negative samples,
which gives our model the best generalization capability on unseen test data.
We will formulate this as a constrained optimization problem.
We want to minimize 1/2 of the square of the magnitude of w, subject to the constraints
that each positive sample is on the positive side of the street and each negative sample is on the negative side.
To solve this constrained optimization problem, we will introduce Lagrange Multipliers (alpha).
This allows us to write the Lagrangian formulation L, and find its extremum by taking partial derivatives.
Setting derivatives to zero reveals that the optimal weight vector w is a linear combination of the training samples.
Furthermore, the optimization depends only on dot products of the samples, enabling the famous Kernel Trick.
"""

MOCK_CHUNKS = [
    {
        "index": 0,
        "start_time": 0.0,
        "end_time": 60.0,
        "text": "Patrick Winston introduces decision boundaries and support vector machines to divide classification spaces."
    },
    {
        "index": 1,
        "start_time": 60.0,
        "end_time": 120.0,
        "text": "He explains how to maximize the margin, calling it the widest street approach to avoid being too close to samples."
    },
    {
        "index": 2,
        "start_time": 120.0,
        "end_time": 180.0,
        "text": "He formulates the optimization problem to minimize 1/2 of the magnitude of w squared, introducing Lagrange Multipliers."
    }
]

MOCK_CONCEPTS_REPORT = """
### Segment 1 (0.0s - 60.0s)
* **Support Vector Machines (SVM)**: Supervised learning models used for binary classification.
* **Decision Boundary**: Separating hyperplane dividing space between classes.

### Segment 2 (60.0s - 120.0s)
* **Margin (Widest Street)**: The geometric distance between the decision hyperplane and closest training examples.
* **Support Vectors**: Training points lying exactly on the gutters of the margin.
"""

MOCK_STRUCTURED_MAP = """
- **Support Vector Machine (SVM)**: A binary classification model that draws a separating hyperplane.
- **Margin Optimization**: Maximizing the width of the separating corridor ('widest street') between classes.
- **Support Vectors**: The training data samples that define the margins ('gutters').
- **Lagrange Multipliers**: Multipliers used to handle inequality constraints in the optimization problem.
"""

MOCK_OUTLINE = [
    {
        "chapter_id": 1,
        "title": "Introduction to Support Vector Machines",
        "approx_timestamps": "0s - 60s",
        "concepts": ["SVM", "decision boundary"]
    },
    {
        "chapter_id": 2,
        "title": "The Widest Street Approach (Margin Maximization)",
        "approx_timestamps": "60s - 120s",
        "concepts": ["margin", "support vectors", "gutters"]
    }
]

MOCK_RESEARCH_REPORT = """
# Verification Reference Guide: Support Vector Machines
* **Generalization**: Maximizing the margin prevents overfitting and ensures robust generalization.
* **Lagrangian Dual**: Converting the primal constrained optimization problem into the dual allows optimization to depend solely on dot products ($x_i \\cdot x_j$).
"""

def run_ollama_diagnostics(model_name: str) -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        req = urllib.request.Request(f"{base_url}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            models = [m["name"] for m in data.get("models", [])]
    except Exception as e:
        print(f"Ollama server not responding: {e}")
        sys.exit(1)

    selected_model = None
    matched = [m for m in models if model_name.lower() in m.lower()]
    if matched:
        selected_model = matched[0]
    else:
        gemma_models = [m for m in models if "gemma" in m.lower()]
        if gemma_models:
            selected_model = gemma_models[0]
        elif models:
            selected_model = models[0]
        else:
            print("No models pulled on Ollama.")
            sys.exit(1)

    payload = {
        "model": selected_model,
        "prompt": "Respond with exactly one word: 'READY'.",
        "stream": False,
        "options": {"num_ctx": 2048}
    }
    try:
        req = urllib.request.Request(
            f"{base_url}/api/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=120) as response:
            json.loads(response.read().decode())
    except Exception as e:
        print(f"Diagnostics benchmark generation failed: {e}")
        sys.exit(1)

    return selected_model

def test_extract_transcript_node(state_base: PipelineState) -> dict:
    state = state_base.copy()
    try:
        res = nodes.extract_text_node(state)
        print("✅ extract_text_node completed successfully!")
        return res
    except Exception as e:
        print("❌ extract_text_node failed:", e)
        raise

def test_guardrail_node(state_base: PipelineState) -> dict:
    state = state_base.copy()
    state["transcript"] = MOCK_TRANSCRIPT
    try:
        res = nodes.guardrail_node(state)
        print("✅ guardrail_node completed successfully!")
        return res
    except Exception as e:
        print(f"❌ guardrail_node failed: {e}")
        raise

def test_map_concepts_node(state_base: PipelineState) -> dict:
    state = state_base.copy()
    state["chunks"] = MOCK_CHUNKS
    try:
        res = nodes.map_concepts_node(state)
        print("✅ map_concepts_node completed successfully!")
        return res
    except Exception as e:
        print(f"❌ map_concepts_node failed: {e}")
        raise

def test_shuffle_concepts_node(state_base: PipelineState) -> dict:
    state = state_base.copy()
    state["mapped_concepts_report"] = MOCK_CONCEPTS_REPORT
    try:
        res = nodes.shuffle_concepts_node(state)
        print("✅ shuffle_concepts_node completed successfully!")
        return res
    except Exception as e:
        print(f"❌ shuffle_concepts_node failed: {e}")
        raise

def test_plan_outline_node(state_base: PipelineState) -> dict:
    state = state_base.copy()
    state["structured_response_map"] = MOCK_STRUCTURED_MAP
    try:
        res = nodes.plan_outline_node(state)
        print("✅ plan_outline_node completed successfully!")
        return res
    except Exception as e:
        print(f"❌ plan_outline_node failed: {e}")
        raise

def test_research_node(state_base: PipelineState) -> dict:
    state = state_base.copy()
    state["chunks"] = MOCK_CHUNKS
    state["structured_response_map"] = MOCK_STRUCTURED_MAP
    state["iteration_count"] = 0
    try:
        res = nodes.research_node(state)
        print("✅ research_node completed successfully!")
        return res
    except Exception as e:
        print(f"❌ research_node failed: {e}")
        raise

def test_write_draft_node(state_base: PipelineState) -> dict:
    state = state_base.copy()
    state["chunks"] = MOCK_CHUNKS
    state["outline"] = MOCK_OUTLINE
    state["research_report"] = MOCK_RESEARCH_REPORT
    try:
        res = nodes.write_draft_node(state)
        print("✅ write_draft_node completed successfully!")
        return res
    except Exception as e:
        print(f"❌ write_draft_node failed: {e}")
        raise

def test_verify_draft_node(state_base: PipelineState) -> dict:
    state = state_base.copy()
    state["draft"] = """
    # Introduction to Support Vector Machines
    Support Vector Machines (SVMs) are tools to classify data. They work by drawing a line to separate classes.
    We try to maximize the margin, which Patrick Winston calls the 'widest street approach'.
    """
    try:
        res = nodes.verify_draft_node(state)
        print("✅ verify_draft_node completed successfully!")
        return res
    except Exception as e:
        print(f"❌ verify_draft_node failed: {e}")
        raise

def test_check_quality_node() -> None:
    state_satisfied = {
        "satisfied": True,
        "iteration_count": 0,
        "node_logs": []
    }
    state_unsatisfied = {
        "satisfied": False,
        "gaps": "Explain gutters more clearly",
        "iteration_count": 0,
        "node_logs": []
    }
    try:
        res1 = nodes.check_quality_node(state_satisfied)
        assert res1.get("next_action") == "save_outputs"
        res2 = nodes.check_quality_node(state_unsatisfied)
        assert res2.get("next_action") == "research"
        print("✅ check_quality_node completed successfully!")
    except Exception as e:
        print(f"❌ check_quality_node failed: {e}")
        raise

def main() -> None:
    parser = argparse.ArgumentParser(description="Test Ollama and isolate individual graph nodes.")
    parser.add_argument("--model", type=str, default="gemma2")
    parser.add_argument("--url", type=str, default="https://www.youtube.com/watch?v=_PwhiWxHK8o")
    parser.add_argument("--only-diagnostics", action="store_true")
    parser.add_argument("--node", type=str, choices=[
        "extract", "guardrail", "map", "shuffle", "outline", "research", "write_draft", "verify_draft", "quality"
    ])
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if not args.only_diagnostics and not args.node and not args.all:
        parser.print_help()
        sys.exit(0)

    resolved_model = run_ollama_diagnostics(args.model)
    if args.only_diagnostics:
        sys.exit(0)

    state_base: PipelineState = {
        "source_input": args.url,
        "source_type": "youtube",
        "source_label": "YouTube Video",
        "provider": "ollama",
        "model": resolved_model,
        "api_key": None,
        "is_informative": None,
        "rejection_message": None,
        "transcript": "",
        "global_context": "",
        "chunks": [],
        "mapped_concepts_report": "",
        "structured_response_map": "",
        "outline": [],
        "current_chapter_idx": 0,
        "research_report": "",
        "research_feedback": None,
        "iteration_count": 0,
        "next_action": "research",
        "draft": "",
        "final": "",
        "satisfied": False,
        "gaps": "",
        "node_logs": []
    }

    try:
        if args.all or args.node == "extract":
            test_extract_transcript_node(state_base)
        if args.all or args.node == "guardrail":
            test_guardrail_node(state_base)
        if args.all or args.node == "map":
            test_map_concepts_node(state_base)
        if args.all or args.node == "shuffle":
            test_shuffle_concepts_node(state_base)
        if args.all or args.node == "outline":
            test_plan_outline_node(state_base)
        if args.all or args.node == "research":
            test_research_node(state_base)
        if args.all or args.node == "write_draft":
            test_write_draft_node(state_base)
        if args.all or args.node == "verify_draft":
            test_verify_draft_node(state_base)
        if args.all or args.node == "quality":
            test_check_quality_node()
        print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
    except Exception as e:
        print(f"❌ Test run failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
