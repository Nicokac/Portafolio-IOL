import json
import logging
from pathlib import Path

import pytest

from shared.cache import visual_cache_registry
from shared.visual_cache_prewarm import prewarm_visual_cache, resolve_top_datasets


class _StubStreamlit:
    def __init__(self) -> None:
        self.session_state: dict[str, object] = {}


def _write_telemetry_sample(path: Path, rows: list[tuple[str, str]]) -> None:
    header = [
        "timestamp",
        "metric_name",
        "duration_ms",
        "status",
        "context",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        handle.write(",".join(header) + "\n")
        for timestamp, dataset in rows:
            context = json.dumps({"dataset_hash": dataset})
            values = [
                timestamp,
                "portfolio.visual_cache",
                "1.00",
                "ok",
                context,
            ]
            handle.write(",".join(values) + "\n")


def test_resolve_top_datasets_prefers_recent(tmp_path: Path) -> None:
    telemetry_file = tmp_path / "metrics.csv"
    _write_telemetry_sample(
        telemetry_file,
        [
            ("2025-10-17T05:18:28.130669+00:00", "alpha"),
            ("2025-10-17T05:19:04.785647+00:00", "alpha"),
            ("2025-10-18T01:00:00+00:00", "beta"),
            ("2025-10-18T01:00:01+00:00", "gamma"),
            ("2025-10-18T02:00:00+00:00", "beta"),
        ],
    )

    result = resolve_top_datasets(
        max_preload_count=3,
        telemetry_files=(telemetry_file,),
    )

    assert result == ["beta", "alpha", "gamma"]


def test_prewarm_visual_cache_populates_store(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    fake_st = _StubStreamlit()
    monkeypatch.setattr("shared.visual_cache_prewarm.st", fake_st)
    monkeypatch.setattr(
        "shared.visual_cache_prewarm.resolve_top_datasets",
        lambda **_: ["hash-a", "hash-b"],
    )

    telemetry_events: list[dict[str, object]] = []

    def fake_log(**kwargs):
        telemetry_events.append(kwargs)

    monkeypatch.setattr("shared.visual_cache_prewarm.log_default_telemetry", fake_log)

    caplog.set_level(logging.INFO)

    visual_cache_registry.reset()
    try:
        datasets = prewarm_visual_cache(force=True)
    finally:
        visual_cache_registry.reset()

    assert datasets == ["hash-a", "hash-b"]

    store = fake_st.session_state.get("__visual_cache_prewarm__")
    assert isinstance(store, dict)
    assert set(store.keys()) == {"hash-a", "hash-b"}
    for entry in store.values():
        components = entry.get("components", {})
        assert components["table"]["status"] == "prefetched"
        assert components["charts"]["status"] == "prefetched"

    assert fake_st.session_state.get("__visual_cache_prewarmed__") is True
    assert fake_st.session_state.get("__visual_cache_prewarm_targets__") == [
        "hash-a",
        "hash-b",
    ]

    assert telemetry_events, "Se esperaba registrar telemetría del precalentamiento"
    event = telemetry_events[-1]
    assert event["phase"] == "portfolio.visual_cache"
    assert "visual_cache_prewarm_ms" in event["extra"]

    assert any("[Warmup]" in message for message in caplog.messages)


def test_prewarm_visual_cache_runs_once(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_st = _StubStreamlit()
    monkeypatch.setattr("shared.visual_cache_prewarm.st", fake_st)

    resolve_calls = 0

    def fake_resolve(**_kwargs):
        nonlocal resolve_calls
        resolve_calls += 1
        return ["hash-x"]

    monkeypatch.setattr("shared.visual_cache_prewarm.resolve_top_datasets", fake_resolve)
    monkeypatch.setattr(
        "shared.visual_cache_prewarm.log_default_telemetry",
        lambda **_: None,
    )

    visual_cache_registry.reset()
    try:
        first = prewarm_visual_cache(force=True)
        second = prewarm_visual_cache()
    finally:
        visual_cache_registry.reset()

    assert first == ["hash-x"]
    assert second == []
    assert resolve_calls == 1
