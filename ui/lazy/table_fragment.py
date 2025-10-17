"""Fragment helpers for the portfolio table."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from .runtime import FragmentContext, lazy_fragment


@contextmanager
def table_fragment(*, dataset_token: str | None = None) -> Iterator[FragmentContext]:
    """Provide a rerun-isolated context for the portfolio table."""

    with lazy_fragment(
        "portfolio_table",
        component="table",
        dataset_token=dataset_token,
    ) as context:
        yield context
