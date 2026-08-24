from aster_row_support.eval_checks import (
    check_expect,
    expected_tool_args,
    load_cases,
    normalize_case,
)


def test_flat_tool_arguments_map_to_lookup_order():
    parsed = expected_tool_args({"order_id": "ORD-1007"})
    assert parsed == ("lookup_order", {"order_id": "ORD-1007"})


def test_check_tool_arguments_accepts_normalized_id():
    expect = {"tool": "order_lookup", "tool_arguments": {"order_id": "ORD-1007"}}
    trace = {"tools_called": [{"name": "lookup_order", "args": {"order_id": " ord-1007 "}}]}
    assert check_expect(expect, "The order is shipped with UPS.", trace) == []


def test_optional_sanitized_lookup_does_not_require_tool():
    expect = {
        "tool": "optional_sanitized_lookup",
        "must_not_include": ["ava.morgan@example.test"],
        "must_refuse_to_disclose": ["email"],
    }
    response = "I cannot share the customer's email or other internal data."
    assert check_expect(expect, response, {"tools_called": []}) == []


def test_must_not_silently_choose_one():
    expect = {"must_not_silently_choose_one": True, "handoff": True}
    fail = check_expect(expect, "The tumbler is dishwasher safe.", {"tools_called": []})
    assert any("Silently chose" in item for item in fail)
    ok = check_expect(
        expect,
        "Current official sources conflict. Please contact support for confirmation.",
        {"tools_called": []},
    )
    assert ok == []


def test_normalize_legacy_content_field():
    case = normalize_case({"id": "x", "content": "Hello"})
    assert case["messages"] == [{"role": "user", "content": "Hello"}]


def test_load_visible_and_custom_cases():
    cases = load_cases(["evaluation/visible-cases.json", "evaluation/custom-cases.json"])
    ids = {c["id"] for c in cases}
    assert "standard-return-window" in ids
    assert "custom-trailplus-shipping" in ids
    assert all("messages" in c for c in cases)
    assert len(cases) == 20
