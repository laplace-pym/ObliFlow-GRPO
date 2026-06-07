from obli_flow.action_parser import parse_response


def test_parse_alfworld_action():
    parsed = parse_response("<think>need apple</think><action>pick up apple from table</action>", "alfworld/AlfredTWEnv")
    assert parsed.valid_format
    assert parsed.action_type == "pick"
    assert "apple" in parsed.action


def test_parse_webshop_search():
    parsed = parse_response("<think>search</think><action>search[black wireless mouse]</action>", "Webshop")
    assert parsed.valid_format
    assert parsed.action_type == "search"
    assert parsed.action_args["query"] == "black wireless mouse"


def test_missing_tags_is_invalid():
    parsed = parse_response("search[black wireless mouse]", "Webshop")
    assert not parsed.valid_format
    assert parsed.action_type == "invalid"
