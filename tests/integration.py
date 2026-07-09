#!/usr/bin/env python3

import argparse
import os
import sys
import time
import socket
from dotenv import load_dotenv

socket.setdefaulttimeout(300)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

def _hr(char: str = "═", width: int = 72) -> str:
    return char * width

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CLI test for the DemystifyAI LangGraph pipeline."
    )
    parser.add_argument("--url", required=True, help="Full YouTube video URL")
    parser.add_argument("--provider", default="groq", choices=["groq", "gemini", "openai", "ollama"])
    parser.add_argument("--model", default="llama-3.3-70b-versatile")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--no-diagram", action="store_true")

    args = parser.parse_args()
    md_lines = []

    def md_append(line: str):
        md_lines.append(line)
        print(line)

    print(_hr())
    md_append(_hr())
    print("  🎥  DemystifyAI")
    md_append("  🎥  DemystifyAI")
    print(_hr())
    md_append(_hr())
    print(f"  URL      : {args.url}")
    md_append(f"  URL      : {args.url}")
    print(f"  Provider : {args.provider}")
    md_append(f"  Provider : {args.provider}")
    print(f"  Model    : {args.model}")
    md_append(f"  Model    : {args.model}")
    print(_hr())
    md_append(_hr())

    wall_start = time.time()

    try:
        from pipeline import run_pipeline

        result = run_pipeline(
            source_input=args.url,
            source_type="youtube",
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            export_diagram=not args.no_diagram,
        )

        total_time = time.time() - wall_start

        print("\n" + _hr())
        md_append(_hr())
        print("  🏁  PIPELINE FINISHED")
        md_append("  🏁  PIPELINE FINISHED")
        print(_hr())
        md_append(_hr())
        print(f"  Total wall-clock time : {total_time:.1f}s\n")
        md_append(f"  Total wall-clock time : {total_time:.1f}s\n")

        if not result.get("is_informative", True):
            print("  🚫  GUARDRAIL REJECTED")
            md_append("  🚫  GUARDRAIL REJECTED")
            print("  " + "\n  ".join(result.get("rejection_message", "").splitlines()))
            md_append("  " + "\n  ".join(result.get("rejection_message", "").splitlines()))
            sys.exit(0)

        loops = result.get("iteration_count", 0) + 1
        print(f"  Guardrail     : ✅ INFORMATIVE")
        md_append(f"  Guardrail     : ✅ INFORMATIVE")
        print(f"  Research loops: {loops}")
        md_append(f"  Research loops: {loops}")
        print()
        md_append("")

        print("  Output files:")
        md_append("  Output files:")
        for fname in [result["explanation_file"]]:
            if os.path.exists(fname):
                size = os.path.getsize(fname)
                line = f"    ✅  {fname:<40s}  ({size:,} bytes)"
                print(line)
                md_append(line)
            else:
                line = f"    ❌  {fname}  — NOT FOUND"
                print(line)
                md_append(line)

        if not args.no_diagram:
            for dname in ["architecture_diagram.png", "architecture_diagram.mmd"]:
                if os.path.exists(dname):
                    size = os.path.getsize(dname)
                    line = f"    ✅  {dname:<40s}  ({size:,} bytes)"
                    print(line)
                    md_append(line)

        if result.get("node_logs"):
            print(f"\n  Node execution log:")
            md_append("\n  Node execution log:")
            for entry in result["node_logs"]:
                print(f"    {entry}")
                md_append(f"    {entry}")

        print("\n" + _hr("─"))
        md_append("\n" + _hr("─"))
        print("  📄  DEMYSTIFIED EXPLANATION — first 600 chars:")
        md_append("  📄  DEMYSTIFIED EXPLANATION — first 600 chars:")
        print(_hr("─"))
        md_append(_hr("─"))
        print(result["explanation_text"][:600])
        md_append(result["explanation_text"][:600])

        print("\n" + _hr())
        md_append("\n" + _hr())

        full_md_lines = []
        full_md_lines.append("\n" + _hr("═"))
        full_md_lines.append(f"🧪  TEST RUN CONTEXT — {time.strftime('%Y-%m-%d %H:%M:%S')}")
        full_md_lines.append(f"  URL      : {args.url}")
        full_md_lines.append(f"  Provider : {args.provider}")
        full_md_lines.append(f"  Model    : {args.model}")
        full_md_lines.append(_hr("═") + "\n")
        full_md_lines.extend(md_lines)
        
        full_md_lines.append("\n" + _hr("─"))
        full_md_lines.append("📝  FULL DEMYSTIFIED EXPLANATION OUTPUT")
        full_md_lines.append(_hr("─"))
        full_md_lines.append(result.get("explanation_text", ""))
        full_md_lines.append("\n" + _hr("═") + "\n\n")

        output_path = os.path.join(os.path.dirname(__file__), "test_output.md")
        with open(output_path, "a", encoding="utf-8") as f_md:
            f_md.write("\n".join(full_md_lines))
        
        root_output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_output.md")
        with open(root_output_path, "a", encoding="utf-8") as f_md:
            f_md.write("\n".join(full_md_lines))

        print(f"  📄  Test output appended to {output_path}")
        print(f"  📄  Test output appended to {root_output_path}")
        sys.exit(0)

    except KeyboardInterrupt:
        print("\n\n⚡  Interrupted by user.")
        sys.exit(1)
    except Exception as exc:
        print(f"\n❌  Pipeline failed: {exc}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
