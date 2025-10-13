from controllers.recommendations_controller import resolve_predictive_spinner


def test_spinner_message_for_running_job() -> None:
    message = resolve_predictive_spinner({"status": "running", "job_id": "abc123"})
    assert message is not None
    assert "abc123" in message
    assert "en segundo plano" in message


def test_spinner_message_for_failed_job() -> None:
    message = resolve_predictive_spinner({"status": "failed"})
    assert message == "No se pudieron actualizar las predicciones sectoriales."


def test_spinner_returns_none_when_idle() -> None:
    assert resolve_predictive_spinner({"status": "finished"}) is None
    assert resolve_predictive_spinner(None) is None
