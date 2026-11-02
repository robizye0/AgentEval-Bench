from agenteval.evaluators.mcp import MCPEvaluator


def test_mcp_schema_match_accepts_equivalent_tool_definition():
    expected = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    actual = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
    assert MCPEvaluator.compare_schema("web_search", expected, actual) == []


def test_mcp_schema_match_reports_nested_constraints_and_enum_drift():
    expected = {
        "type": "object",
        "properties": {
            "filter": {
                "type": "object",
                "properties": {"state": {"type": "string", "enum": ["open", "closed"]}},
                "required": ["state"],
            },
            "limit": {"type": "integer", "maximum": 50},
        },
        "required": ["filter", "limit"],
    }
    actual = {
        "type": "object",
        "properties": {
            "filter": {
                "type": "object",
                "properties": {"state": {"type": "string", "enum": ["open", "archived"]}},
                "required": [],
            },
            "limit": {"type": "integer", "maximum": 100},
        },
        "required": ["filter", "limit"],
    }
    differences = MCPEvaluator.compare_schema("query", expected, actual)
    assert any("enum" in item for item in differences)
    assert any("nested required fields differ" in item for item in differences)
    assert any("maximum" in item for item in differences)


def test_mcp_schema_match_reports_required_type_and_missing_field_drift():
    expected = {
        "type": "object",
        "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
        "required": ["query", "limit"],
    }
    actual = {"type": "object", "properties": {"query": {"type": "number"}}, "required": []}
    differences = MCPEvaluator.compare_schema("web_search", expected, actual)
    assert any("missing property 'limit'" in item for item in differences)
    assert any("type mismatch for 'query'" in item for item in differences)
    assert any("required fields differ" in item for item in differences)
