import csv

from shared import qa_profiler
from shared import telemetry


def test_profiler_writes_metrics_snapshot(tmp_path, monkeypatch):
    qa_profiler._reset_for_tests()
    monkeypatch.setattr(telemetry, "_QA_METRICS_FILE", tmp_path / "qa_metrics.csv", raising=False)

    qa_profiler.record_startup_complete()
    with qa_profiler.track_ui_render():
        pass
    with qa_profiler.track_cache_load():
        pass
    with qa_profiler.track_auth_latency():
        pass

    metrics_path = telemetry._QA_METRICS_FILE
    assert metrics_path.exists()

    with metrics_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows, "expected QA profiler to persist at least one row"
    last_row = rows[-1]

    assert list(last_row.keys()) == list(telemetry._QA_METRIC_COLUMNS)

    for field in telemetry._QA_METRIC_COLUMNS[2:]:
        value = last_row[field]
        assert value != ""
        float(value)


def test_telemetry_log_appends_metrics(tmp_path, monkeypatch):
    qa_profiler._reset_for_tests()
    target = tmp_path / "qa_metrics.csv"
    monkeypatch.setattr(telemetry, "_QA_METRICS_FILE", target, raising=False)

    telemetry.log(
        "qa_manual",
        qa=True,
        startup_time_ms=12.5,
        ui_render_time_ms=7.2,
        cache_load_time_ms=3.4,
        auth_latency_ms=8.8,
    )

    with target.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows
    row = rows[0]
    assert row["startup_time_ms"] == "12.500"
    assert row["ui_render_time_ms"] == "7.200"
    assert row["cache_load_time_ms"] == "3.400"
    assert row["auth_latency_ms"] == "8.800"
