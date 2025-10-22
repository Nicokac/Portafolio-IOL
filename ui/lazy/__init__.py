"""Lazy rendering helpers for isolating Streamlit reruns."""

from .charts_fragment import charts_fragment
from .runtime import (
    FragmentContext,
    current_component,
    current_scope,
    in_form_scope,
    lazy_fragment,
)
from .table_fragment import table_fragment

__all__ = [
    "FragmentContext",
    "current_component",
    "current_scope",
    "in_form_scope",
    "lazy_fragment",
    "table_fragment",
    "charts_fragment",
]
