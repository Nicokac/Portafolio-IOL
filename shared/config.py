# shared/config.py
from __future__ import annotations
import os, json
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional
from dotenv import load_dotenv
from functools import lru_cache
import logging
from logging.handlers import TimedRotatingFileHandler
import sys
import streamlit as st

try:  # pragma: no cover - import may fail in tests
    from streamlit.runtime.secrets import StreamlitSecretNotFoundError
except Exception:  # pragma: no cover - streamlit may not expose runtime module
    class StreamlitSecretNotFoundError(Exception):
        """Fallback cuando streamlit no expone StreamlitSecretNotFoundError."""
        pass

logger = logging.getLogger(__name__)

# Raíz del proyecto (donde están app.py, .env, config.json, etc.)
BASE_DIR = Path(__file__).resolve().parents[1]

# Cargar variables del .env en la raíz (y fallback al cwd por si acaso)
load_dotenv(BASE_DIR / ".env")
load_dotenv()

DEFAULT_LOG_RETENTION_DAYS = 7


class _PatternLevelOverrideFilter(logging.Filter):
    """Filter que re-nivela mensajes que matchean un patrón dado."""

    def __init__(self, pattern: re.Pattern[str], level: int) -> None:
        super().__init__()
        self._pattern = pattern
        self._level = level

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - simple guard
        try:
            message = record.getMessage()
        except Exception:
            message = str(record.msg)
        if self._pattern.search(str(message)):
            record.levelno = self._level
            record.levelname = logging.getLevelName(self._level)
        return True


def _downgrade_logger_patterns(
    logger_name: str, patterns: Iterable[str], *, level: int = logging.DEBUG
) -> None:
    compiled = [p for p in (re.compile(pattern) for pattern in patterns) if p]
    if not compiled:
        return
    logger = logging.getLogger(logger_name)
    for pattern in compiled:
        logger.addFilter(_PatternLevelOverrideFilter(pattern, level))

def _load_cfg() -> Dict[str, Any]:
    """
    Carga (opcional) config.json desde la raíz del proyecto (o cwd). Si no existe, {}.
    """
    candidates = [BASE_DIR / "config.json", Path.cwd() / "config.json"]
    for p in candidates:
        try:
            if p.exists():
                return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.exception("No se pudo cargar configuración %s: %s", p, e)
    return {}

class Settings:
    def __init__(self) -> None:
        cfg = _load_cfg()

        # --- Identidad / headers ---
        self.USER_AGENT: str = os.getenv("USER_AGENT", cfg.get("USER_AGENT", "IOL-Portfolio/1.0 (+app)"))

        # --- Servicios auxiliares ---
        self.NOTIFICATIONS_URL: str | None = self.secret_or_env(
            "NOTIFICATIONS_URL", cfg.get("NOTIFICATIONS_URL")
        )
        self.NOTIFICATIONS_TIMEOUT: float = float(
            os.getenv("NOTIFICATIONS_TIMEOUT", cfg.get("NOTIFICATIONS_TIMEOUT", 3.0))
        )

        # --- Macro data providers ---
        self.MACRO_API_PROVIDER: str = os.getenv(
            "MACRO_API_PROVIDER", cfg.get("MACRO_API_PROVIDER", "fred")
        )
        raw_series = self.secret_or_env("FRED_SECTOR_SERIES", cfg.get("FRED_SECTOR_SERIES"))
        self.FRED_SECTOR_SERIES: Dict[str, str] = self._parse_sector_series(raw_series)
        raw_fallback = self.secret_or_env(
            "MACRO_SECTOR_FALLBACK", cfg.get("MACRO_SECTOR_FALLBACK")
        )
        self.MACRO_SECTOR_FALLBACK: Dict[str, Dict[str, Any]] = self._parse_macro_fallback(
            raw_fallback
        )
        self.FRED_API_KEY: str | None = self.secret_or_env(
            "FRED_API_KEY", cfg.get("FRED_API_KEY")
        )
        self.FRED_API_BASE_URL: str = os.getenv(
            "FRED_API_BASE_URL",
            cfg.get("FRED_API_BASE_URL", "https://api.stlouisfed.org/fred"),
        )
        self.FRED_API_RATE_LIMIT_PER_MINUTE: int = int(
            os.getenv(
                "FRED_API_RATE_LIMIT_PER_MINUTE",
                cfg.get("FRED_API_RATE_LIMIT_PER_MINUTE", 120),
            )
        )
        raw_worldbank_series = self.secret_or_env(
            "WORLD_BANK_SECTOR_SERIES", cfg.get("WORLD_BANK_SECTOR_SERIES")
        )
        self.WORLD_BANK_SECTOR_SERIES: Dict[str, str] = self._parse_sector_series(
            raw_worldbank_series
        )
        self.WORLD_BANK_API_KEY: str | None = self.secret_or_env(
            "WORLD_BANK_API_KEY", cfg.get("WORLD_BANK_API_KEY")
        )
        self.WORLD_BANK_API_BASE_URL: str = os.getenv(
            "WORLD_BANK_API_BASE_URL",
            cfg.get("WORLD_BANK_API_BASE_URL", "https://api.worldbank.org/v2"),
        )
        self.WORLD_BANK_API_RATE_LIMIT_PER_MINUTE: int = int(
            os.getenv(
                "WORLD_BANK_API_RATE_LIMIT_PER_MINUTE",
                cfg.get("WORLD_BANK_API_RATE_LIMIT_PER_MINUTE", 60),
            )
        )

        # --- Financial Modeling Prep ---
        self.FMP_API_KEY: str | None = self.secret_or_env(
            "FMP_API_KEY", cfg.get("FMP_API_KEY")
        )
        self.FMP_BASE_URL: str = os.getenv(
            "FMP_BASE_URL",
            cfg.get("FMP_BASE_URL", "https://financialmodelingprep.com/api/v3"),
        )
        self.FMP_TIMEOUT: float = float(
            os.getenv("FMP_TIMEOUT", cfg.get("FMP_TIMEOUT", 5.0))
        )

        # --- Credenciales IOL ---
        self.IOL_USERNAME: str | None = self.secret_or_env("IOL_USERNAME", cfg.get("IOL_USERNAME"))
        self.IOL_PASSWORD: str | None = self.secret_or_env("IOL_PASSWORD", cfg.get("IOL_PASSWORD"))

        # --- Logging ---
        retention_candidate = os.getenv(
            "LOG_RETENTION_DAYS", cfg.get("LOG_RETENTION_DAYS", DEFAULT_LOG_RETENTION_DAYS)
        )
        self.LOG_RETENTION_DAYS: int = self._coerce_positive_int(retention_candidate)
        self.SQLITE_MAINTENANCE_INTERVAL_HOURS: float = float(
            os.getenv(
                "SQLITE_MAINTENANCE_INTERVAL_HOURS",
                cfg.get("SQLITE_MAINTENANCE_INTERVAL_HOURS", 6.0),
            )
        )
        self.SQLITE_MAINTENANCE_SIZE_THRESHOLD_MB: float = float(
            os.getenv(
                "SQLITE_MAINTENANCE_SIZE_THRESHOLD_MB",
                cfg.get("SQLITE_MAINTENANCE_SIZE_THRESHOLD_MB", 256.0),
            )
        )
        self.PERFORMANCE_STORE_TTL_DAYS: float = float(
            os.getenv(
                "PERFORMANCE_STORE_TTL_DAYS",
                cfg.get("PERFORMANCE_STORE_TTL_DAYS", self.LOG_RETENTION_DAYS),
            )
        )

        # --- Cache/TTLs usados en app.py ---
        self.cache_ttl_portfolio: int = int(
            os.getenv("CACHE_TTL_PORTFOLIO", cfg.get("CACHE_TTL_PORTFOLIO", 3600))
        )
        self.cache_ttl_last_price: int = int(os.getenv("CACHE_TTL_LAST_PRICE", cfg.get("CACHE_TTL_LAST_PRICE", 10)))
        self.cache_ttl_fx: int = int(os.getenv("CACHE_TTL_FX", cfg.get("CACHE_TTL_FX", 60)))
        self.cache_ttl_quotes: int = int(
            os.getenv("CACHE_TTL_QUOTES", cfg.get("CACHE_TTL_QUOTES", 600))
        )
        self.cache_ttl_yf_indicators: int = int(
            os.getenv("CACHE_TTL_YF_INDICATORS", cfg.get("CACHE_TTL_YF_INDICATORS", 900))
        )
        self.cache_ttl_yf_history: int = int(
            os.getenv("CACHE_TTL_YF_HISTORY", cfg.get("CACHE_TTL_YF_HISTORY", 3600))
        )
        self.cache_ttl_yf_fundamentals: int = int(
            os.getenv("CACHE_TTL_YF_FUNDAMENTALS", cfg.get("CACHE_TTL_YF_FUNDAMENTALS", 21600))
        )
        self.cache_ttl_yf_portfolio_fundamentals: int = int(
            os.getenv(
                "CACHE_TTL_YF_PORTFOLIO_FUNDAMENTALS",
                cfg.get("CACHE_TTL_YF_PORTFOLIO_FUNDAMENTALS", 14400),
            )
        )
        self.MARKET_DATA_CACHE_BACKEND: str = str(
            os.getenv(
                "MARKET_DATA_CACHE_BACKEND",
                cfg.get("MARKET_DATA_CACHE_BACKEND", "sqlite"),
            )
            or "sqlite"
        ).strip().lower()
        self.MARKET_DATA_CACHE_PATH: str | None = os.getenv(
            "MARKET_DATA_CACHE_PATH",
            cfg.get("MARKET_DATA_CACHE_PATH", "data/market_cache.db"),
        )
        self.MARKET_DATA_CACHE_REDIS_URL: str | None = os.getenv(
            "MARKET_DATA_CACHE_REDIS_URL",
            cfg.get("MARKET_DATA_CACHE_REDIS_URL"),
        )
        self.MARKET_DATA_CACHE_TTL: float = float(
            os.getenv("MARKET_DATA_CACHE_TTL", cfg.get("MARKET_DATA_CACHE_TTL", 6 * 60 * 60))
        )
        self.quotes_hist_maxlen: int = int(os.getenv("QUOTES_HIST_MAXLEN", cfg.get("QUOTES_HIST_MAXLEN", 500)))
        self.max_quote_workers: int = int(os.getenv("MAX_QUOTE_WORKERS", cfg.get("MAX_QUOTE_WORKERS", 12)))
        default_quotes_batch = cfg.get("QUOTES_BATCH_SIZE", max(self.max_quote_workers, 8))
        self.quotes_batch_size: int = int(
            os.getenv("QUOTES_BATCH_SIZE", default_quotes_batch)
        )
        default_swr_ttl = cfg.get("QUOTES_SWR_TTL_SECONDS", 30.0)
        self.quotes_swr_ttl_seconds: float = float(
            os.getenv("QUOTES_SWR_TTL_SECONDS", default_swr_ttl)
        )
        default_swr_grace = cfg.get(
            "QUOTES_SWR_GRACE_SECONDS", float(self.cache_ttl_quotes)
        )
        self.quotes_swr_grace_seconds: float = float(
            os.getenv("QUOTES_SWR_GRACE_SECONDS", default_swr_grace)
        )
        self.YAHOO_FUNDAMENTALS_TTL: int = int(
            os.getenv("YAHOO_FUNDAMENTALS_TTL", cfg.get("YAHOO_FUNDAMENTALS_TTL", 3600))
        )
        self.YAHOO_QUOTES_TTL: int = int(
            os.getenv("YAHOO_QUOTES_TTL", cfg.get("YAHOO_QUOTES_TTL", 300))
        )
        self.YAHOO_REQUEST_DELAY: float = float(
            os.getenv("YAHOO_REQUEST_DELAY", cfg.get("YAHOO_REQUEST_DELAY", 0.0))
        )
        self.QUOTES_TTL_SECONDS: int = int(
            os.getenv("QUOTES_TTL_SECONDS", cfg.get("QUOTES_TTL_SECONDS", 300))
        )
        self.QUOTES_RPS_IOL: float = float(
            os.getenv("QUOTES_RPS_IOL", cfg.get("QUOTES_RPS_IOL", 3))
        )
        self.QUOTES_RPS_LEGACY: float = float(
            os.getenv("QUOTES_RPS_LEGACY", cfg.get("QUOTES_RPS_LEGACY", 1))
        )
        self.LEGACY_LOGIN_MAX_RETRIES: int = int(
            os.getenv(
                "LEGACY_LOGIN_MAX_RETRIES",
                cfg.get("LEGACY_LOGIN_MAX_RETRIES", 1),
            )
        )
        self.LEGACY_LOGIN_BACKOFF_BASE: float = float(
            os.getenv(
                "LEGACY_LOGIN_BACKOFF_BASE",
                cfg.get("LEGACY_LOGIN_BACKOFF_BASE", 0.5),
            )
        )

        self.min_score_threshold: int = int(
            os.getenv("MIN_SCORE_THRESHOLD", cfg.get("MIN_SCORE_THRESHOLD", 80))
        )
        self.max_results: int = int(os.getenv("MAX_RESULTS", cfg.get("MAX_RESULTS", 20)))
        self.RISK_BADGE_THRESHOLD: float = float(
            os.getenv("RISK_BADGE_THRESHOLD", cfg.get("RISK_BADGE_THRESHOLD", 0.75))
        )
        self.TECHNICAL_SIGNAL_THRESHOLD: float = float(
            os.getenv(
                "TECHNICAL_SIGNAL_THRESHOLD",
                cfg.get("TECHNICAL_SIGNAL_THRESHOLD", 2),
            )
        )
        self.EARNINGS_UPCOMING_DAYS: int = int(
            os.getenv("EARNINGS_UPCOMING_DAYS", cfg.get("EARNINGS_UPCOMING_DAYS", 7))
        )

        self.STUB_MAX_RUNTIME_WARN: float = float(
            os.getenv("STUB_MAX_RUNTIME_WARN", cfg.get("STUB_MAX_RUNTIME_WARN", 0.25))
        )

        # --- Snapshots backend configuration ---
        self.snapshot_backend: str = os.getenv(
            "SNAPSHOT_BACKEND", cfg.get("SNAPSHOT_BACKEND", "json")
        )
        self.snapshot_storage_path: str | None = os.getenv(
            "SNAPSHOT_STORAGE_PATH", cfg.get("SNAPSHOT_STORAGE_PATH")
        )
        raw_retention = os.getenv("SNAPSHOT_RETENTION", cfg.get("SNAPSHOT_RETENTION"))
        try:
            self.snapshot_retention: int | None = (
                int(raw_retention) if raw_retention not in (None, "") else None
            )
        except (TypeError, ValueError):
            logger.warning("Valor inválido para SNAPSHOT_RETENTION: %s", raw_retention)
            self.snapshot_retention = None

        primary_raw = os.getenv(
            "OHLC_PRIMARY_PROVIDER", cfg.get("OHLC_PRIMARY_PROVIDER", "alpha_vantage")
        )
        primary_text = str(primary_raw or "").strip().lower()
        self.OHLC_PRIMARY_PROVIDER: str = primary_text or "alpha_vantage"

        raw_secondary = self.secret_or_env(
            "OHLC_SECONDARY_PROVIDERS", cfg.get("OHLC_SECONDARY_PROVIDERS")
        )
        fallback_secondaries = [] if self.OHLC_PRIMARY_PROVIDER == "polygon" else ["polygon"]
        parsed_secondaries = self._parse_provider_list(
            raw_secondary,
            fallback=fallback_secondaries,
        )
        self.OHLC_SECONDARY_PROVIDERS: list[str] = [
            provider for provider in parsed_secondaries if provider != self.OHLC_PRIMARY_PROVIDER
        ]

        self.ALPHA_VANTAGE_API_KEY: str | None = self.secret_or_env(
            "ALPHA_VANTAGE_API_KEY", cfg.get("ALPHA_VANTAGE_API_KEY")
        )
        self.ALPHA_VANTAGE_BASE_URL: str = os.getenv(
            "ALPHA_VANTAGE_BASE_URL",
            cfg.get("ALPHA_VANTAGE_BASE_URL", "https://www.alphavantage.co/query"),
        )
        self.POLYGON_API_KEY: str | None = self.secret_or_env(
            "POLYGON_API_KEY", cfg.get("POLYGON_API_KEY")
        )
        self.POLYGON_BASE_URL: str = os.getenv(
            "POLYGON_BASE_URL", cfg.get("POLYGON_BASE_URL", "https://api.polygon.io")
        )

        # --- Archivo de tokens (IOLAuth) ---
        # Por defecto lo guardamos en la raíz junto a app.py (compat con tu tokens_iol.json existente)
        self.tokens_file: str = self.secret_or_env(
            "IOL_TOKENS_FILE", cfg.get("IOL_TOKENS_FILE", str(BASE_DIR / "tokens_iol.json"))
        )
        # Clave opcional para cifrar/descifrar el archivo de tokens (Fernet)
        raw_tokens_key = self.secret_or_env("IOL_TOKENS_KEY", cfg.get("IOL_TOKENS_KEY"))
        if raw_tokens_key:
            normalized_tokens_key = str(raw_tokens_key).strip()
            self.tokens_key = normalized_tokens_key if normalized_tokens_key else None
        else:
            self.tokens_key = None

        raw_fastapi_key = self.secret_or_env("FASTAPI_TOKENS_KEY", cfg.get("FASTAPI_TOKENS_KEY"))
        if raw_fastapi_key:
            normalized_fastapi_key = str(raw_fastapi_key).strip()
            self.fastapi_tokens_key = normalized_fastapi_key if normalized_fastapi_key else None
        else:
            self.fastapi_tokens_key = None

        if self.fastapi_tokens_key and self.tokens_key and self.fastapi_tokens_key == self.tokens_key:
            raise RuntimeError(
                "FASTAPI_TOKENS_KEY must be different from IOL_TOKENS_KEY to isolate encryption scopes."
            )

        # Permite (opcionalmente) guardar tokens sin cifrar si falta tokens_key
        self.allow_plain_tokens: bool = (
            os.getenv("IOL_ALLOW_PLAIN_TOKENS", str(cfg.get("IOL_ALLOW_PLAIN_TOKENS", ""))).lower()
            in ("1", "true", "yes")
        )
        raw_app_env = os.getenv("APP_ENV", cfg.get("APP_ENV", "dev"))
        app_env_text = str(raw_app_env or "dev").strip()
        self.app_env: str = app_env_text.lower() or "dev"
        # TTL máximo para reutilizar tokens guardados (en días)
        self.tokens_ttl_days: int = int(
            os.getenv("IOL_TOKENS_TTL_DAYS", cfg.get("IOL_TOKENS_TTL_DAYS", 30))
        )

        # --- Derivados de dólar (Ahorro/Tarjeta a partir del oficial) ---
        self.fx_ahorro_multiplier: float = float(os.getenv("FX_AHORRO_MULTIPLIER", cfg.get("FX_AHORRO_MULTIPLIER", 1.30)))
        self.fx_tarjeta_multiplier: float = float(os.getenv("FX_TARJETA_MULTIPLIER", cfg.get("FX_TARJETA_MULTIPLIER", 1.35)))

        # --- Logging ---
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", cfg.get("LOG_LEVEL", "INFO")).upper()
        self.LOG_FORMAT: str = os.getenv("LOG_FORMAT", cfg.get("LOG_FORMAT", "plain")).lower()
        self.REDIS_URL: str | None = self.secret_or_env("REDIS_URL", cfg.get("REDIS_URL"))
        self.ENABLE_PROMETHEUS: bool = str(
            os.getenv("ENABLE_PROMETHEUS", cfg.get("ENABLE_PROMETHEUS", "1"))
        ).lower() in {"1", "true", "yes"}
        self.PERFORMANCE_VERBOSE_TEXT_LOG: bool = str(
            os.getenv(
                "PERFORMANCE_VERBOSE_TEXT_LOG",
                cfg.get("PERFORMANCE_VERBOSE_TEXT_LOG", "0"),
            )
        ).lower() in {"1", "true", "yes"}

    def secret_or_env(self, key: str, default: Any | None = None) -> Any | None:
        try:
            return st.secrets[key]
        except (KeyError, StreamlitSecretNotFoundError, AttributeError):
            return os.getenv(key, default)

    @staticmethod
    def _parse_jsonish(raw: Any) -> Any:
        if raw is None:
            return None
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return raw
        return raw

    def _parse_provider_list(
        self, raw: Any, *, fallback: Iterable[str] | None = None
    ) -> list[str]:
        parsed = self._parse_jsonish(raw)
        if isinstance(parsed, str):
            candidates_iter: Iterable[str] = [
                item.strip() for item in parsed.split(",") if item.strip()
            ]
        elif isinstance(parsed, Iterable) and not isinstance(parsed, (bytes, bytearray, str)):
            candidates_iter = parsed
        else:
            candidates_iter = []

        normalized: list[str] = []
        for item in candidates_iter:
            name = str(item or "").strip().lower()
            if name and name not in normalized:
                normalized.append(name)

        if not normalized and fallback is not None:
            for item in fallback:
                name = str(item or "").strip().lower()
                if name and name not in normalized:
                    normalized.append(name)
        return normalized

    def _parse_sector_series(self, raw: Any) -> Dict[str, str]:
        parsed = self._parse_jsonish(raw)
        if not isinstance(parsed, Mapping):
            return {}
        mapping: Dict[str, str] = {}
        for key, value in parsed.items():
            label = str(key or "").strip()
            if not label:
                continue
            series_id: str = ""
            if isinstance(value, Mapping):
                series_id = str(
                    value.get("series_id")
                    or value.get("series")
                    or value.get("id")
                    or ""
                ).strip()
            else:
                series_id = str(value or "").strip()
            if not series_id:
                continue
            mapping[label] = series_id
        return mapping

    def _parse_macro_fallback(self, raw: Any) -> Dict[str, Dict[str, Any]]:
        parsed = self._parse_jsonish(raw)
        if not isinstance(parsed, Mapping):
            return {}
        fallback: Dict[str, Dict[str, Any]] = {}
        for key, value in parsed.items():
            label = str(key or "").strip()
            if not label:
                continue
            numeric_value: Optional[float] = None
            as_of: Optional[str] = None
            if isinstance(value, Mapping):
                raw_value = value.get("value")
                try:
                    numeric_value = float(raw_value)
                except (TypeError, ValueError):
                    continue
                raw_as_of = value.get("as_of")
                if raw_as_of is not None:
                    text = str(raw_as_of).strip()
                    if text:
                        as_of = text
            else:
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    continue
            entry: Dict[str, Any] = {"value": numeric_value}
            if as_of:
                entry["as_of"] = as_of
            fallback[label] = entry
        return fallback

    def _coerce_positive_int(self, candidate: Any) -> int:
        try:
            value = int(candidate)
        except (TypeError, ValueError):
            return DEFAULT_LOG_RETENTION_DAYS
        return max(value, 1)

settings = Settings()


def ensure_tokens_key() -> None:
    """Verifica que exista una clave para cifrar tokens.

    Si falta ``IOL_TOKENS_KEY`` y no se habilitó ``IOL_ALLOW_PLAIN_TOKENS``,
    se registra un error y se aborta la ejecución con ``sys.exit(1)``.
    """

    if not settings.tokens_key and not settings.allow_plain_tokens:
        logger.error(
            "IOL_TOKENS_KEY no está configurada y IOL_ALLOW_PLAIN_TOKENS no está habilitado."
        )
        sys.exit(1)


class JsonFormatter(logging.Formatter):
    """Formato JSON simple para registros de log."""

    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        user = os.getenv("LOG_USER")
        if user:
            log_record["user"] = user
        return json.dumps(log_record)


LOG_FILENAME_PATTERN = re.compile(r"analysis_(\d{4}-\d{2}-\d{2})\.log$")


def prune_old_logs(directory: Path, retention_days: int, current_file: str | None = None) -> None:
    """Remove ``analysis_YYYY-MM-DD.log`` files older than the retention window."""

    try:
        retention_value = int(retention_days)
    except (TypeError, ValueError):
        retention_value = DEFAULT_LOG_RETENTION_DAYS

    retention_value = max(retention_value, 1)
    cutoff = datetime.now().date() - timedelta(days=retention_value - 1)

    current_path = Path(current_file) if current_file else None
    current_resolved = current_path.resolve() if current_path else None

    legacy_file = directory / "analysis.log"
    if legacy_file.exists():
        try:
            legacy_file.unlink()
        except OSError as exc:
            logger.warning("No se pudo borrar log legacy %s: %s", legacy_file, exc)

    for candidate in directory.glob("analysis_*.log"):
        if current_resolved is not None and candidate.resolve() == current_resolved:
            continue

        match = LOG_FILENAME_PATTERN.match(candidate.name)
        if not match:
            continue

        try:
            file_date = datetime.strptime(match.group(1), "%Y-%m-%d").date()
        except ValueError:
            continue

        if file_date < cutoff:
            try:
                candidate.unlink()
            except OSError as exc:
                logger.warning("No se pudo borrar log antiguo %s: %s", candidate, exc)


class DailyTimedRotatingFileHandler(TimedRotatingFileHandler):
    """Time-based file handler that writes daily log files with a date suffix."""

    def __init__(self, directory: Path, retention_days: int, encoding: str = "utf-8") -> None:
        self.log_directory = Path(directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)
        try:
            retention_value = int(retention_days)
        except (TypeError, ValueError):
            retention_value = DEFAULT_LOG_RETENTION_DAYS
        self.retention_days = max(retention_value, 1)

        current_time = datetime.now()
        filename = self._filename_for(current_time)

        super().__init__(
            filename=str(filename),
            when="midnight",
            interval=1,
            backupCount=0,
            encoding=encoding,
            delay=False,
        )

        # Ensure the next rollover happens at the upcoming midnight regardless of
        # the file's modification time (especially when the file is freshly
        # created during startup).
        self.rolloverAt = self.computeRollover(time.time())

        prune_old_logs(self.log_directory, self.retention_days, current_file=self.baseFilename)

    def _filename_for(self, moment: datetime) -> Path:
        return self.log_directory / f"analysis_{moment.strftime('%Y-%m-%d')}.log"

    def doRollover(self) -> None:  # pragma: no cover - exercised indirectly
        if self.stream:
            self.stream.close()
            self.stream = None

        rollover_time = self.rolloverAt or time.time()
        next_moment = datetime.fromtimestamp(rollover_time)
        self.baseFilename = str(self._filename_for(next_moment))

        if not self.delay:
            self.stream = self._open()

        # Schedule the subsequent rollover and prune stale files.
        self.rolloverAt = self.computeRollover(rollover_time)
        prune_old_logs(self.log_directory, self.retention_days, current_file=self.baseFilename)


def configure_logging(level: str | None = None, json_format: bool | None = None) -> None:
    """Configura el logging global.

    Por defecto usa nivel ``INFO`` y formato ``"plain"``. Los valores
    configurados se normalizan y, si son inválidos, se revierte a estos
    predeterminados. Los parámetros permiten sobrescribir el nivel y el
    formato configurados mediante variables de entorno.
    """

    level_name = (level or getattr(settings, "LOG_LEVEL", "INFO")).upper()
    level_value = getattr(logging, level_name, None)
    if not isinstance(level_value, int):
        level_name = "INFO"
        level_value = logging.INFO

    if json_format is None:
        fmt = os.getenv("LOG_FORMAT", getattr(settings, "LOG_FORMAT", "plain"))
        fmt = str(fmt).lower()
        if fmt not in {"json", "plain"}:
            fmt = "plain"
        json_format = fmt == "json"

    if json_format:
        formatter: logging.Formatter = JsonFormatter(datefmt="%Y-%m-%d %H:%M:%S")
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    root = logging.getLogger()
    root.setLevel(level_value)
    root.handlers = []

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    log_directory = BASE_DIR
    try:
        log_directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    retention_days = getattr(settings, "LOG_RETENTION_DAYS", DEFAULT_LOG_RETENTION_DAYS)
    try:
        retention_days = int(retention_days)
    except (TypeError, ValueError):
        retention_days = DEFAULT_LOG_RETENTION_DAYS
    retention_days = max(retention_days, 1)

    prune_old_logs(log_directory, retention_days)

    file_handler = DailyTimedRotatingFileHandler(
        directory=log_directory,
        retention_days=retention_days,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root.addHandler(stream_handler)
    root.addHandler(file_handler)

    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

    kaleido_noise_levels: Mapping[str, int] = {
        "kaleido": logging.WARNING,
        "kaleido.scopes": logging.WARNING,
        "kaleido.scopes.base": logging.ERROR,
        "kaleido.scopes.plotly": logging.WARNING,
        "plotly.io._base_renderers": logging.WARNING,
        "plotly.io._kaleido": logging.WARNING,
        "Choreographer": logging.ERROR,
        "choreographer": logging.ERROR,
    }

    for logger_name, forced_level in kaleido_noise_levels.items():
        logging.getLogger(logger_name).setLevel(forced_level)

    for target in (
        "yfinance",
        "yfinance.scrapers",
        "yfinance.ticker",
        "yfinance.data",
    ):
        _downgrade_logger_patterns(target, (r"(?i)404", r"(?i)not found"))

@lru_cache(maxsize=1)
def get_config() -> dict:
    path = os.getenv("PORTFOLIO_CONFIG_PATH", "config.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f) or {}
            if not isinstance(cfg.get("cedear_to_us", {}), dict):
                cfg["cedear_to_us"] = {}
            if not isinstance(cfg.get("etfs", []), list):
                cfg["etfs"] = []
            if not isinstance(cfg.get("acciones_ar", []), list):
                cfg["acciones_ar"] = []
            if not isinstance(cfg.get("fci_symbols", []), list):
                cfg["fci_symbols"] = []
            if not isinstance(cfg.get("scale_overrides", {}), dict):
                cfg["scale_overrides"] = {}
            if not isinstance(cfg.get("classification_patterns", {}), dict):
                cfg["classification_patterns"] = {}
            return cfg
    except FileNotFoundError:
        logger.warning("No se encontró archivo de configuración: %s", path)
        return {}
    except (OSError, json.JSONDecodeError) as e:
        logger.exception("Error cargando configuración %s: %s", path, e)
        return {}
