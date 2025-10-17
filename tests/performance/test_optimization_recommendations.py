from pathlib import Path


def test_performance_plan_exists_and_mentions_objective():
    plan_path = Path("performance_optimization_plan.md")
    assert plan_path.is_file(), "El archivo performance_optimization_plan.md debe existir"
    content = plan_path.read_text(encoding="utf-8")
    assert "Plan de optimización" in content
    assert "<10 s" in content or "10 s" in content, "El plan debe mencionar la meta de 10 segundos"
