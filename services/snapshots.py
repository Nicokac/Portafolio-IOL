"""Snapshot storage helpers with pluggable backends.

This module centralises the logic required to persist and retrieve
snapshots of the different analytical views (portafolio, técnico,
riesgo).  Two storage implementations are available out-of-the-box:

* JSON files: simple append-only structure suited for local usage.
* SQLite: transactional backend suitable for concurrent readers.

The active backend can be selected through configuration or via
``configure_storage`` at runtime.  Public helpers expose a small API
used by controllers and services to persist snapshots, list historical
records and compute comparisons between arbitrary entries.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
import json
import logging
import os
import sys
from pathlib import Path
import sqlite3
import threading
import time
import uuid
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

from services import health
from shared.snapshot import compress_payload, decompress_payload

logger = logging.getLogger(__name__)

_CONFIG_LOCK = threading.RLock()
_STORAGE: _BaseSnapshotStorage | None = None  # Lazily configured backend instance


def auto_configure_if_needed() -> None:
    """Ensure a usable backend is configured when running without settings."""

    global _STORAGE

    with _CONFIG_LOCK:
        storage = _STORAGE
        needs_configuration = storage is None or getattr(storage, "backend_name", None) == "unconfigured"
        if not needs_configuration:
            return
        try:
            configure_storage(backend="json")
        except Exception:  # pragma: no cover - defensive logging path
            logger.exception("No se pudo autoconfigurar el backend de snapshots")



def _ensure_configured() -> None:
    """Garantiza que el backend de snapshots esté listo antes de utilizarlo."""

    global _STORAGE

    if _STORAGE is not None:
        return

    with _CONFIG_LOCK:
        if _STORAGE is not None:
            return

        try:
            _auto_configure_from_settings()
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.exception("Fallo al configurar backend de snapshots: %s", exc)
            raise


SnapshotPayload = Mapping[str, Any]
SnapshotMetadata = Mapping[str, Any]


class SnapshotStorageError(RuntimeError):
    """Raised when the snapshot backend cannot fulfil an operation."""


class _BaseSnapshotStorage:
    """Common interface for snapshot storage backends."""

    def save_snapshot(
        self, kind: str, payload: SnapshotPayload, metadata: SnapshotMetadata | None = None
    ) -> Mapping[str, Any]:
        raise NotImplementedError

    def load_snapshot(self, snapshot_id: str) -> Mapping[str, Any] | None:
        raise NotImplementedError

    def list_snapshots(
        self, kind: str | None = None, *, limit: int | None = None, order: str = "desc"
    ) -> Sequence[Mapping[str, Any]]:
        raise NotImplementedError


class _NullSnapshotStorage(_BaseSnapshotStorage):
    """No-op backend used as safe fallback when configuration fails."""

    def save_snapshot(self, kind: str, payload: SnapshotPayload, metadata: SnapshotMetadata | None = None):
        logger.debug("NullSnapshotStorage.save_snapshot called for kind=%s", kind)
        now = time.time()
        return {
            "id": str(uuid.uuid4()),
            "kind": kind,
            "created_at": now,
            "payload": dict(payload or {}),
            "metadata": dict(metadata or {}),
        }

    def load_snapshot(self, snapshot_id: str):  # pragma: no cover - defensive default
        logger.debug("NullSnapshotStorage.load_snapshot called for id=%s", snapshot_id)
        return None

    def list_snapshots(self, kind: str | None = None, *, limit: int | None = None, order: str = "desc"):
        logger.debug("NullSnapshotStorage.list_snapshots called for kind=%s", kind)
        return []


class _JSONSnapshotStorage(_BaseSnapshotStorage):
    """Persist snapshots in a JSON document stored on disk."""

    def __init__(self, path: Path, *, retention: int | None = None) -> None:
        self.path = path
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_permissions()
        self._retention: int | None = None
        if retention is not None:
            try:
                retention_value = int(retention)
            except (TypeError, ValueError):
                logger.warning("Valor de retención inválido %s, se ignorará", retention)
            else:
                if retention_value > 0:
                    self._retention = retention_value
                elif retention_value != 0:
                    logger.warning(
                        "La retención de snapshots debe ser positiva, se ignorará: %s",
                        retention,
                    )

    def _ensure_permissions(self) -> None:
        parent = self.path.parent
        if not os.access(parent, os.W_OK | os.R_OK):
            raise SnapshotStorageError(f"Permisos insuficientes en {parent}")
        if self.path.exists() and not os.access(self.path, os.W_OK | os.R_OK):
            raise SnapshotStorageError(f"No se puede leer/escribir {self.path}")

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
                if isinstance(raw, list):
                    for row in raw:
                        if isinstance(row, dict):
                            row["payload"] = decompress_payload(row.get("payload"))
                    return raw
                logger.warning("Formato inesperado en %s, se esperaba lista", self.path)
                return []
        except json.JSONDecodeError:
            logger.exception("JSON inválido en %s", self.path)
            raise SnapshotStorageError(f"Contenido inválido en {self.path}")
        except OSError as err:
            logger.exception("No se pudo leer %s: %s", self.path, err)
            raise SnapshotStorageError(str(err)) from err

    def _dump(self, rows: Sequence[Mapping[str, Any]]) -> None:
        tmp_path = self.path.with_suffix(".tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as fh:
                json.dump(list(rows), fh, ensure_ascii=False)
                fh.flush()
                os.fsync(fh.fileno())
            tmp_path.replace(self.path)
        except OSError as err:
            logger.exception("No se pudo escribir %s: %s", self.path, err)
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError as cleanup_err:  # pragma: no cover - best effort cleanup
                logger.warning(
                    "No se pudo eliminar archivo temporal %s tras fallo: %s",
                    tmp_path,
                    cleanup_err,
                )
            raise SnapshotStorageError(str(err)) from err

    def save_snapshot(self, kind: str, payload: SnapshotPayload, metadata: SnapshotMetadata | None = None):
        entry = {
            "id": str(uuid.uuid4()),
            "kind": str(kind or "").strip() or "generic",
            "created_at": time.time(),
            "payload": compress_payload(_to_plain_mapping(payload)),
            "metadata": _to_plain_mapping(metadata),
        }
        with self._lock:
            rows = self._load()
            rows.append(entry)
            rows_to_persist: Sequence[Mapping[str, Any]] = rows
            if self._retention and len(rows) > self._retention:
                rows_to_persist = rows[-self._retention :]
            self._dump(rows_to_persist)
        entry["payload"] = decompress_payload(entry["payload"])
        return entry

    def load_snapshot(self, snapshot_id: str):
        if not snapshot_id:
            return None
        with self._lock:
            rows = self._load()
        for row in rows:
            if row.get("id") == snapshot_id:
                return row
        return None

    def list_snapshots(self, kind: str | None = None, *, limit: int | None = None, order: str = "desc"):
        with self._lock:
            rows = list(self._load())
        if kind:
            rows = [row for row in rows if row.get("kind") == kind]
        rows.sort(key=lambda row: row.get("created_at", 0.0), reverse=order.lower() != "asc")
        if limit is not None and limit >= 0:
            rows = rows[:limit]
        return rows


class _SQLiteSnapshotStorage(_BaseSnapshotStorage):
    """Persist snapshots using a SQLite database."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_permissions()
        self._init_db()

    def _ensure_permissions(self) -> None:
        parent = self.path.parent
        if not os.access(parent, os.W_OK | os.R_OK):
            raise SnapshotStorageError(f"Permisos insuficientes en {parent}")
        if self.path.exists() and not os.access(self.path, os.W_OK | os.R_OK):
            raise SnapshotStorageError(f"No se puede leer/escribir {self.path}")

    def _init_db(self) -> None:
        try:
            with sqlite3.connect(self.path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS snapshots (
                        id TEXT PRIMARY KEY,
                        kind TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        payload TEXT NOT NULL,
                        metadata TEXT
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_snapshots_kind_created_at ON snapshots(kind, created_at)"
                )
        except sqlite3.Error as err:
            logger.exception("No se pudo inicializar SQLite en %s: %s", self.path, err)
            raise SnapshotStorageError(str(err)) from err

    def save_snapshot(self, kind: str, payload: SnapshotPayload, metadata: SnapshotMetadata | None = None):
        compressed_payload = compress_payload(_to_plain_mapping(payload))
        entry = {
            "id": str(uuid.uuid4()),
            "kind": str(kind or "").strip() or "generic",
            "created_at": time.time(),
            "payload": json.dumps(compressed_payload, ensure_ascii=False),
            "metadata": json.dumps(_to_plain_mapping(metadata), ensure_ascii=False),
        }
        try:
            with sqlite3.connect(self.path) as conn:
                conn.execute(
                    "INSERT INTO snapshots (id, kind, created_at, payload, metadata) VALUES (?, ?, ?, ?, ?)",
                    (
                        entry["id"],
                        entry["kind"],
                        entry["created_at"],
                        entry["payload"],
                        entry["metadata"],
                    ),
                )
        except sqlite3.Error as err:
            logger.exception("No se pudo guardar snapshot %s: %s", kind, err)
            raise SnapshotStorageError(str(err)) from err
        entry["payload"] = decompress_payload(json.loads(entry["payload"]))
        entry["metadata"] = json.loads(entry["metadata"]) if entry["metadata"] else {}
        return entry

    def load_snapshot(self, snapshot_id: str):
        if not snapshot_id:
            return None
        try:
            with sqlite3.connect(self.path) as conn:
                cur = conn.execute(
                    "SELECT id, kind, created_at, payload, metadata FROM snapshots WHERE id = ?",
                    (snapshot_id,),
                )
                row = cur.fetchone()
        except sqlite3.Error as err:
            logger.exception("No se pudo leer snapshot %s: %s", snapshot_id, err)
            raise SnapshotStorageError(str(err)) from err
        if not row:
            return None
        payload = decompress_payload(json.loads(row[3]) if row[3] else {})
        metadata = json.loads(row[4]) if row[4] else {}
        return {
            "id": row[0],
            "kind": row[1],
            "created_at": row[2],
            "payload": payload,
            "metadata": metadata,
        }

    def list_snapshots(self, kind: str | None = None, *, limit: int | None = None, order: str = "desc"):
        params: list[Any] = []
        where = ""
        if kind:
            where = "WHERE kind = ?"
            params.append(kind)
        order_clause = "DESC" if order.lower() != "asc" else "ASC"
        limit_clause = ""
        if limit is not None and limit >= 0:
            limit_clause = "LIMIT ?"
            params.append(limit)
        sql = f"SELECT id, kind, created_at, payload, metadata FROM snapshots {where} ORDER BY created_at {order_clause} {limit_clause}"
        try:
            with sqlite3.connect(self.path) as conn:
                cur = conn.execute(sql, tuple(params))
                rows = cur.fetchall()
        except sqlite3.Error as err:
            logger.exception("No se pudo listar snapshots %s: %s", kind or "*", err)
            raise SnapshotStorageError(str(err)) from err
        records = []
        for row in rows:
            payload = decompress_payload(json.loads(row[3]) if row[3] else {})
            metadata = json.loads(row[4]) if row[4] else {}
            records.append(
                {
                    "id": row[0],
                    "kind": row[1],
                    "created_at": row[2],
                    "payload": payload,
                    "metadata": metadata,
                }
            )
        return records


def current_backend_name() -> str:
    """Return the identifier for the active snapshot backend."""

    auto_configure_if_needed()

    storage = _STORAGE
    if storage is None:
        return "unconfigured"

    if isinstance(storage, _JSONSnapshotStorage):
        return "json"
    if isinstance(storage, _SQLiteSnapshotStorage):
        return "sqlite"
    if isinstance(storage, _NullSnapshotStorage):
        return "null"
    return type(storage).__name__


def is_null_backend() -> bool:
    """Return whether the current backend disables snapshot persistence."""

    if _STORAGE is None:
        return True
    return isinstance(_STORAGE, _NullSnapshotStorage)


def configure_storage(
    *,
    backend: str | None = None,
    path: str | os.PathLike[str] | None = None,
    retention: int | None = None,
) -> None:
    """Configure the snapshot backend based on the provided parameters."""

    backend = (backend or "json").strip().lower()
    storage_path = Path(path) if path else None
    retention_value: int | None = None
    if retention is not None:
        try:
            parsed_retention = int(retention)
        except (TypeError, ValueError):
            logger.warning("Valor inválido para retention=%s, se ignorará", retention)
        else:
            if parsed_retention > 0:
                retention_value = parsed_retention
            elif parsed_retention != 0:
                logger.warning(
                    "La retención de snapshots debe ser positiva, se ignorará: %s",
                    retention,
                )

    try:
        if backend == "json":
            storage = _JSONSnapshotStorage(
                storage_path or Path("data/snapshots.json"), retention=retention_value
            )
        elif backend in {"sqlite", "sqlite3"}:
            storage = _SQLiteSnapshotStorage(storage_path or Path("data/snapshots.db"))
        elif backend in {"null", "none", "disabled"}:
            storage = _NullSnapshotStorage()
        else:
            raise SnapshotStorageError(f"Backend no soportado: {backend}")
    except SnapshotStorageError as err:
        logger.exception("Fallo la configuración del backend de snapshots (%s)", backend)
        storage = _NullSnapshotStorage()

        detail_text = str(err).strip() or repr(err)
        metadata = {"backend": backend}
        if storage_path is not None:
            metadata["path"] = str(storage_path)

        try:
            health.record_risk_incident(
                category="snapshots.backend",
                severity="error",
                detail=f"No se pudo inicializar el backend '{backend}': {detail_text}",
                fallback=True,
                source="snapshots.configure_storage",
                tags=("snapshots", backend),
                metadata=metadata,
            )
        except Exception:  # pragma: no cover - defensive fallback
            logger.exception("No se pudo registrar la incidencia de snapshots")

    global _STORAGE
    with _CONFIG_LOCK:
        _STORAGE = storage


def save_snapshot(kind: str, payload: SnapshotPayload, metadata: SnapshotMetadata | None = None) -> Mapping[str, Any]:
    """Persist a snapshot and return the stored record."""

    _ensure_configured()
    storage = _STORAGE
    if storage is None:  # pragma: no cover - defensive guard
        raise SnapshotStorageError("Backend de snapshots no configurado")

    payload = _to_plain_mapping(payload)
    metadata = _to_plain_mapping(metadata)
    try:
        return storage.save_snapshot(kind, payload, metadata)
    except SnapshotStorageError:
        logger.exception("Error al guardar snapshot %s", kind)
        raise


def load_snapshot(snapshot_id: str) -> Mapping[str, Any] | None:
    """Load a snapshot by identifier."""

    _ensure_configured()
    storage = _STORAGE
    if storage is None:  # pragma: no cover - defensive guard
        raise SnapshotStorageError("Backend de snapshots no configurado")

    try:
        return storage.load_snapshot(snapshot_id)
    except SnapshotStorageError:
        logger.exception("Error al cargar snapshot %s", snapshot_id)
        raise


def list_snapshots(
    kind: str | None = None, *, limit: int | None = None, order: str = "desc"
) -> Sequence[Mapping[str, Any]]:
    """Return historical snapshots for the requested kind."""

    _ensure_configured()
    storage = _STORAGE
    if storage is None:  # pragma: no cover - defensive guard
        raise SnapshotStorageError("Backend de snapshots no configurado")

    try:
        return storage.list_snapshots(kind, limit=limit, order=order)
    except SnapshotStorageError:
        logger.exception("Error al listar snapshots kind=%s", kind)
        raise


def compare_snapshots(id_a: str, id_b: str) -> Mapping[str, Any] | None:
    """Compute deltas between two stored snapshots.

    The function returns a dictionary including the totals for both
    snapshots and the difference (``a - b``).  ``None`` is returned when
    either snapshot is missing.
    """

    if not id_a or not id_b:
        return None

    _ensure_configured()
    storage = _STORAGE
    if storage is None:  # pragma: no cover - defensive guard
        raise SnapshotStorageError("Backend de snapshots no configurado")

    try:
        return _compare_snapshots_with_storage(storage, id_a, id_b)
    except SnapshotStorageError:
        logger.exception("Error al comparar snapshots %s vs %s", id_a, id_b)
        raise


def _compare_snapshots_with_storage(
    storage: _BaseSnapshotStorage, id_a: str, id_b: str
) -> Mapping[str, Any] | None:
    snap_a = storage.load_snapshot(id_a)
    snap_b = storage.load_snapshot(id_b)
    if not snap_a or not snap_b:
        return None

    totals_a = _extract_totals(snap_a.get("payload"))
    totals_b = _extract_totals(snap_b.get("payload"))
    delta = {
        key: (totals_a.get(key) or 0.0) - (totals_b.get(key) or 0.0)
        for key in sorted({*totals_a.keys(), *totals_b.keys()})
    }

    return {
        "id_a": id_a,
        "id_b": id_b,
        "created_at_a": snap_a.get("created_at"),
        "created_at_b": snap_b.get("created_at"),
        "metadata_a": snap_a.get("metadata") or {},
        "metadata_b": snap_b.get("metadata") or {},
        "totals_a": totals_a,
        "totals_b": totals_b,
        "delta": delta,
    }


def _extract_totals(payload: Mapping[str, Any] | None) -> Dict[str, float]:
    totals = {}
    if not isinstance(payload, Mapping):
        return totals
    raw = payload.get("totals")
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            try:
                totals[key] = float(value)
            except (TypeError, ValueError):
                continue
    return totals


def _to_plain_mapping(data: Mapping[str, Any] | None) -> Dict[str, Any]:
    if not data:
        return {}
    if is_dataclass(data):
        return asdict(data)
    if isinstance(data, Mapping):
        plain: Dict[str, Any] = {}
        for key, value in data.items():
            if is_dataclass(value):
                plain[key] = asdict(value)
            elif isinstance(value, Mapping):
                plain[key] = _to_plain_mapping(value)
            elif isinstance(value, (list, tuple)):
                plain[key] = [_to_plain_mapping(v) if isinstance(v, Mapping) else v for v in value]
            else:
                plain[key] = value
        return plain
    return {"value": data}


def _auto_configure_from_settings() -> None:
    try:
        from shared.settings import settings as _settings

        backend = getattr(_settings, "snapshot_backend", "json")
        path = getattr(_settings, "snapshot_storage_path", None)
        retention = getattr(_settings, "snapshot_retention", None)
        configure_storage(backend=backend, path=path, retention=retention)
    except Exception:  # pragma: no cover - safeguards for early imports
        logger.exception("No se pudo inicializar el backend de snapshots, se usará NullStorage")
        configure_storage(backend="null")


if os.getenv("STREAMLIT_RUNTIME") or "streamlit" in sys.modules:
    auto_configure_if_needed()

