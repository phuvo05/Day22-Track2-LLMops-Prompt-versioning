"""
run_all.py — Run all 4 steps sequentially
==========================================
Executes Task 1 through Task 4 in order.
Usage:
    python run_all.py          # Run all steps
    python run_all.py --step 2 # Run from step 2 onward
    python run_all.py --skip 3 # Skip step 3 (RAGAS, takes ~15-20 min)
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


STEPS = [
    ("Step 1: LangSmith RAG Pipeline", "01_langsmith_rag_pipeline.py"),
    ("Step 2: Prompt Hub & A/B Routing", "02_prompt_hub_ab_routing.py"),
    ("Step 3: RAGAS Evaluation (~15-20 min)", "03_ragas_evaluation.py"),
    ("Step 4: Guardrails AI Validators", "04_guardrails_validator.py"),
]

SKIP_NOTES = {
    1: "Step 1 (LangSmith) — may take ~2-3 min",
    2: "Step 2 (Prompt Hub) — may take ~3-5 min",
    3: "Step 3 (RAGAS) — takes ~15-20 min",
    4: "Step 4 (Guardrails) — takes ~30 sec",
}


def run_step(step_num: int, name: str, script: str) -> bool:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  [{step_num}/4] {name}")
    print(sep)

    script_path = Path(script)
    if not script_path.exists():
        print(f"ERROR: {script} not found. Make sure you are in the project root.")
        return False

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: {script} exited with code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run all Day 22 lab steps sequentially.")
    parser.add_argument(
        "--step", type=int, choices=[1, 2, 3, 4],
        help="Start from this step number (1-4)"
    )
    parser.add_argument(
        "--skip", type=int, choices=[1, 2, 3, 4],
        help="Skip this step number (1-4)"
    )
    args = parser.parse_args()

    # Determine which steps to run
    skip_set = {args.skip} if args.skip else set()
    start_idx = (args.step - 1) if args.step else 0

    steps_to_run = [(i + 1, name, script) for i, (name, script) in enumerate(STEPS) if i >= start_idx and (i + 1) not in skip_set]

    if not steps_to_run:
        print("No steps to run. Check --step and --skip arguments.")
        sys.exit(1)

    # Print plan
    print(f"{'=' * 70}")
    print("  Day 22 Lab — Run All Steps")
    print(f"{'=' * 70}")
    print(f"  Steps to run: {[s[0] for s in steps_to_run]}")
    if args.skip:
        print(f"  Skipping: Step {args.skip} ({SKIP_NOTES.get(args.skip, '')})")
    print(f"{'=' * 70}\n")

    # Confirm for long-running runs (step 3 = RAGAS)
    if any(s[0] == 3 for s in steps_to_run):
        confirm = input("Step 3 (RAGAS) takes ~15-20 minutes. Continue? [y/N]: ").strip().lower()
        if confirm not in ("y", "yes"):
            print("Aborted.")
            sys.exit(0)

    # Run steps
    success = True
    for step_num, name, script in steps_to_run:
        ok = run_step(step_num, name, script)
        if not ok:
            success = False
            break

    # Summary
    print(f"\n{'=' * 70}")
    if success:
        print("  All steps completed successfully!")
    else:
        print("  Some steps failed. Check output above.")
    print(f"{'=' * 70}")
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
