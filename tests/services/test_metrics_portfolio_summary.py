from __future__ import annotations

from collections import deque

import pytest

from services.health import (
    _summarize_portfolio_stats as summarize_portfolio_stats_alias,
)
from services.health.metrics_portfolio import summarize_portfolio_stats


@pytest.fixture()
def sample_portfolio_stats() -> dict[str, object]:
    return {
        "invocations": 5,
        "latency_count": 3,
        "latency_sum": 150.0,
        "latency_sum_sq": 8300.0,
        "latency_min": 30.0,
        "latency_max": 70.0,
        "latency_history": deque([30.0, 50.0, 70.0]),
        "success_count": 5,
        "success_sum": 4.0,
        "success_sum_sq": 3.6,
        "success_min": 0.6,
        "success_max": 1.0,
        "success_history": deque([1.0, 0.8, 0.6, 1.0, 0.6]),
        "sources": {"api": 3, "cache": 2},
        "event_history": deque(
            [
                {"source": "api", "detail": "primary", "ts": 1.0},
                {"source": "cache", "detail": "fallback", "ts": 2.0},
            ]
        ),
    }


def test_default_summary_includes_all_sections(
    sample_portfolio_stats: dict[str, object],
) -> None:
    summary = summarize_portfolio_stats(sample_portfolio_stats)

    assert summary["invocations"] == 5
    assert summary["latency"]["count"] == 3
    assert summary["latency"]["avg"] == pytest.approx(50.0)
    assert summary["latency"]["stdev"] == pytest.approx(16.3299316)
    assert summary["latency"]["min"] == 30.0
    assert summary["latency"]["max"] == 70.0
    assert summary["latency"]["samples"] == [30.0, 50.0, 70.0]
    assert summary["missing_latency"] == 2

    assert summary["success"]["count"] == 5
    assert summary["success"]["avg"] == pytest.approx(0.8)
    assert summary["success"]["stdev"] == pytest.approx(0.2828427)
    assert summary["success"]["min"] == 0.6
    assert summary["success"]["max"] == 1.0

    sources = summary["sources"]
    assert sources["counts"] == {"api": 3, "cache": 2}
    assert sources["ratios"]["api"] == pytest.approx(0.6)
    assert sources["ratios"]["cache"] == pytest.approx(0.4)

    assert summary["events"] == [
        {"source": "api", "detail": "primary", "ts": 1.0},
        {"source": "cache", "detail": "fallback", "ts": 2.0},
    ]


def test_can_skip_success_block(sample_portfolio_stats: dict[str, object]) -> None:
    summary = summarize_portfolio_stats(sample_portfolio_stats, include_success=False)

    assert "success" not in summary
    assert "latency" in summary


def test_can_skip_latency_block(sample_portfolio_stats: dict[str, object]) -> None:
    summary = summarize_portfolio_stats(sample_portfolio_stats, include_latency=False)

    assert "latency" not in summary
    assert "missing_latency" not in summary
    assert "success" in summary


def test_alias_matches_canonical_implementation(
    sample_portfolio_stats: dict[str, object],
) -> None:
    expected = summarize_portfolio_stats(
        sample_portfolio_stats,
        include_success=False,
        include_latency=True,
    )
    alias_result = summarize_portfolio_stats_alias(
        sample_portfolio_stats,
        include_success=False,
        include_latency=True,
    )

    assert alias_result == expected
