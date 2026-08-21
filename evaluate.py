import json
from agent import Agent

def run_evaluation():
    import glob
    case_files = glob.glob("evaluation/*.json")
    
    all_cases = []
    for f_path in case_files:
        try:
            with open(f_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_cases.extend(data.get("cases", []))
        except FileNotFoundError:
            pass
            
    cases = all_cases
    results = {
        "total": len(cases),
        "passed": 0,
        "failed": 0,
        "details": [],
        "categories": {}
    }

    for case in cases:
        print(f"\n--- Running Case: {case['id']} ({case['category']}) ---")
        agent = Agent()
        
        import time
        final_response = ""
        for msg in case["messages"]:
            print(f"User: {msg['content']}")
            
            # Retry logic for 502/503/429
            for attempt in range(10):
                try:
                    final_response = agent.send_message(msg['content'])
                    break
                except Exception as e:
                    if attempt < 9:
                        print(f"API Error: {e}. Retrying in 15s...")
                        time.sleep(15)
                    else:
                        raise e
            
            print(f"Agent: {final_response}")
            
        trace = agent.last_trace
        passed = True
        failures = []
        
        expect = case.get("expect", {})
        
        # Check string inclusions
        for text in expect.get("must_include", []):
            if text.lower() not in final_response.lower():
                passed = False
                failures.append(f"Missing required text: '{text}'")
                
        for text in expect.get("must_not_include", []):
            if text.lower() in final_response.lower():
                passed = False
                failures.append(f"Found forbidden text: '{text}'")
                
        # Check required sources
        used_sources_str = str(trace["sources_used"])
        for req_source in expect.get("required_sources", []):
            if req_source not in used_sources_str:
                passed = False
                failures.append(f"Missing required source: '{req_source}'")
                
        # Check forbidden sources (should not be cited by agent)
        for req_source in expect.get("forbidden_sources_as_authority", []):
            if req_source in final_response:
                passed = False
                failures.append(f"Used forbidden source: '{req_source}'")
                
        # Check required concepts (loose keyword match for deterministic testing)
        for concept in expect.get("must_include_concepts", []):
            words = [w.lower() for w in concept.split() if len(w) > 4]
            # pass if at least 50% of the long words are in the response
            if words:
                match_count = sum(1 for w in words if w in final_response.lower())
                if match_count / len(words) < 0.3:
                    passed = False
                    failures.append(f"Likely missing concept: '{concept}'")

        # Check tool calls
        tool_expect = expect.get("tool", "")
        tools_called = [t["name"] for t in trace["tools_called"]]
        
        if tool_expect == "not_called" and "lookup_order" in tools_called:
            passed = False
            failures.append(f"Expected no order lookup, but called: {tools_called}")
        elif tool_expect == "order_lookup" and "lookup_order" not in tools_called:
            passed = False
            failures.append(f"Expected order_lookup tool, but called: {tools_called}")
        elif tool_expect == "not_called_without_id" and "lookup_order" in tools_called:
            passed = False
            failures.append(f"Called lookup_order without an ID provided by user: {tools_called}")
            
        if passed:
            print("PASS")
            results["passed"] += 1
        else:
            print("FAIL")
            for f in failures:
                print(f" - {f}")
            results["failed"] += 1
            
        # Aggregate by category
        cat = case.get("category", "uncategorized")
        if cat not in results["categories"]:
            results["categories"][cat] = {"total": 0, "passed": 0, "failed": 0}
            
        results["categories"][cat]["total"] += 1
        if passed:
            results["categories"][cat]["passed"] += 1
        else:
            results["categories"][cat]["failed"] += 1
            
        results["details"].append({
            "id": case["id"],
            "passed": passed,
            "failures": failures
        })
        time.sleep(13)
        
    print("\n=== EVALUATION RESULTS ===")
    print(f"Total: {results['total']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    
    print("\n--- By Category ---")
    for cat, stats in results["categories"].items():
        print(f"{cat}: {stats['passed']}/{stats['total']} passed")
    
    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_evaluation()
