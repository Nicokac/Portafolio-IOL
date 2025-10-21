"""Tests for shared health utility helpers."""

from collections import deque

import math

import pytest

from services.health import _serialize_event_history as init_serialize
from services.health import _summarize_metric_block as init_summarize
from services.health.utils import (
    _serialize_event_history as utils_serialize,
    _summarize_metric_block as utils_summarize,
)


def test_health_utils_reexports_are_identical():
    """Ensure the package reexports the canonical utility helpers."""

    assert init_serialize is utils_serialize
    assert init_summarize is utils_summarize


@pytest.mark.parametrize(
    "history",
    [
        [],
        [
            {"ts": 1, "status": "ok"},
            {"ts": 2, "status": "warn", "detail": "slow"},
        ],
        deque([
            {"ts": 3, "status": "error", "detail": "timeout"},
            {"ts": 4, "status": "ok"},
        ]),
    ],
)
def test_serialize_event_history_consistency(history):
    """Both module entry points should serialize history identically."""

    assert init_serialize(history) == utils_serialize(history)


def test_summarize_metric_block_consistency():
    """Both entry points should summarise metric blocks identically."""

    stats = {
        "latency_count": 4,
        "latency_sum": 100.0,
        "latency_sum_sq": 3000.0,
        "latency_min": 10,
        "latency_max": 40,
        "latency_history": deque([10, 20, 30, 40]),
    }

    expected = {
        "count": 4,
        "avg": pytest.approx(25.0),
        "stdev": pytest.approx(math.sqrt(125.0)),
        "min": 10.0,
        "max": 40.0,
        "samples": [10, 20, 30, 40],
    }

    init_summary = init_summarize(stats, "latency")
    utils_summary = utils_summarize(stats, "latency")

    assert init_summary == utils_summary
    assert init_summary == expected


def test_summarize_metric_block_handles_invalid_stats():
    """Invalid or empty stats should produce ``None`` consistently."""

    stats = {"latency_count": 0}

    assert init_summarize(stats, "latency") is None
    assert utils_summarize(stats, "latency") is None

    assert init_summarize(None, "latency") is None
    assert utils_summarize(None, "latency") is None
