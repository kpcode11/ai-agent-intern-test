import glob
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from aster_row_support.agent import Agent
from aster_row_support.eval_checks import check_expect, load_cases

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_evaluation():
    cases = load_cases(sorted(glob.glob(str(PROJECT_ROOT / "evaluation" / "*.json"))))
    results = {
        "total": len(cases),
        "passed": 0,
        "failed": 0,
        "details": [],
        "categories": {},
    }

    for case in cases:
        print(f"\n--- Running Case: {case['id']} ({case.get('category', 'uncategorized')}) ---")
        agent = Agent()

        final_response = ""
        for msg in case["messages"]:
            print(f"User: {msg['content']}")
            for attempt in range(10):
                try:
                    final_response = agent.send_message(msg["content"])
                    break
                except Exception as e:
                    if attempt < 9:
                        print(f"API Error: {e}. Retrying in 15s...")
                        time.sleep(15)
                    else:
                        raise
            print(f"Agent: {final_response}")

        trace = agent.session_trace
        failures = check_expect(case.get("expect", {}), final_response, trace)
        passed = not failures

        if passed:
            print("PASS")
            results["passed"] += 1
        else:
            print("FAIL")
            for item in failures:
                print(f" - {item}")
            results["failed"] += 1

        cat = case.get("category", "uncategorized")
        if cat not in results["categories"]:
            results["categories"][cat] = {"total": 0, "passed": 0, "failed": 0}
        results["categories"][cat]["total"] += 1
        if passed:
            results["categories"][cat]["passed"] += 1
        else:
            results["categories"][cat]["failed"] += 1

        results["details"].append(
            {
                "id": case["id"],
                "category": cat,
                "passed": passed,
                "failures": failures,
            }
        )

    print("\n=== EVALUATION RESULTS ===")
    print(f"Total: {results['total']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")

    print("\n--- By Category ---")
    for cat, stats in results["categories"].items():
        print(f"{cat}: {stats['passed']}/{stats['total']} passed")

    with (PROJECT_ROOT / "eval_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    run_evaluation()
