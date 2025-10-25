from __future__ import annotations

import datetime as _dt
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
CATALOG_PATH = Path("data/assets_catalog.json")


def _ensure_list(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        result: list[dict[str, Any]] = []
        for item in data:
            if isinstance(item, dict):
                result.append(dict(item))
        return result
    if isinstance(data, dict):
        return [
            {"simbolo": str(symbol), "tipo": value}
            for symbol, value in data.items()
        ]
    raise TypeError("El catálogo debe ser una lista de objetos o un mapeo de símbolos")


def _normalize_entry(item: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(item)
    symbol = str(normalized.get("simbolo", ""))
    normalized["simbolo"] = symbol.strip().upper()

    raw_tipo = normalized.get("tipo")
    if raw_tipo is not None:
        normalized["tipo"] = str(raw_tipo)
    prev_standard = normalized.get("tipo_estandar")
    if prev_standard is not None:
        normalized["tipo_estandar"] = str(prev_standard)
    return normalized


def sync_catalog(path: Path = CATALOG_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el catálogo en {path}")

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    entries = _ensure_list(data)
    normalized_entries: list[dict[str, Any]] = []

    for entry in entries:
        normalized_entries.append(_normalize_entry(entry))

    normalized_entries.sort(key=lambda item: item.get("simbolo", ""))
    backup_name = f"assets_catalog_backup_{_dt.date.today().isoformat()}.json"
    backup_path = path.with_name(backup_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(path, backup_path)

    with path.open("w", encoding="utf-8") as fh:
        json.dump(normalized_entries, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")

    return normalized_entries


def main() -> None:
    catalog = sync_catalog()
    total = len(catalog)
    print(f"Catálogo sincronizado ({total} activos normalizados).")


if __name__ == "__main__":
    main()
