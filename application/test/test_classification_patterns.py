from application.portfolio_service import classify_symbol


def test_classify_symbol_uses_config_patterns():
    assert classify_symbol("AL30") == "Bono / ON"
    assert classify_symbol("S10N5") == "Bono / ON"
