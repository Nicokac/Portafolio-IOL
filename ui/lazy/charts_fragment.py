"""Fragment helpers for the portfolio charts."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from .runtime import FragmentContext, lazy_fragment


@contextmanager
def charts_fragment(*, dataset_token: str | None = None) -> Iterator[FragmentContext]:
    """Provide a rerun-isolated context for the portfolio charts."""

    with lazy_fragment(
        "portfolio_charts",
        component="charts",
        dataset_token=dataset_token,
    ) as context:
        yield context
