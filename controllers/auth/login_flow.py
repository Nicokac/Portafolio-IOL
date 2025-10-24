"""Post-login orchestration helpers for authentication flows."""

from __future__ import annotations

import logging
from typing import Any

from services.data_fetch_service import get_portfolio_data_fetch_service

logger = logging.getLogger(__name__)


def force_portfolio_refresh_after_login(client: Any) -> None:
    """Force a portfolio dataset refresh immediately after authentication."""

    if client is None:
        return

    try:
        from controllers.portfolio.portfolio import get_portfolio_service

        service = get_portfolio_data_fetch_service()
        portfolio_service = get_portfolio_service()
        dataset, metadata = service.get_dataset(client, portfolio_service, force_refresh=True)
        logger.info(
            "auth.login_refresh dataset_recomputed",
            extra={
                "auth_refresh_forced": True,
                "dataset_hash": getattr(dataset, "dataset_hash", ""),
                "quotes_hash": getattr(dataset, "quotes_hash", ""),
                "dataset_source": getattr(metadata, "source", "unknown") if metadata else "unknown",
            },
        )
    except Exception:  # pragma: no cover - defensive safeguard
        logger.exception("No se pudo refrescar el dataset tras autenticación")


__all__ = ["force_portfolio_refresh_after_login"]

