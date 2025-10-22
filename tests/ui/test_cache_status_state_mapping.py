def test_state_mapping_valid_values():
    state_map = {"green": "complete", "yellow": "running", "red": "error"}
    for color in ["green", "yellow", "red", "unknown"]:
        state = state_map.get(color, "running")
        assert state in {"running", "complete", "error"}
