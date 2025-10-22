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
from shared.asset_type_aliases import normalize_asset_type  # noqa: E402

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


def _normalize_entry(item: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    normalized = dict(item)
    symbol = str(normalized.get("simbolo", ""))
    normalized["simbolo"] = symbol.strip().upper()

    raw_tipo = normalized.get("tipo")
    if raw_tipo is not None:
        normalized["tipo"] = str(raw_tipo)
    raw_desc = normalized.get("descripcion")
    prev_standard = normalized.get("tipo_estandar")

    standard = normalize_asset_type(str(raw_tipo) if raw_tipo else raw_desc)
    changed = standard != prev_standard

    if standard:
        normalized["tipo_estandar"] = standard
    else:
        normalized.pop("tipo_estandar", None)

    return normalized, changed


def sync_catalog(path: Path = CATALOG_PATH) -> tuple[list[dict[str, Any]], int]:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el catálogo en {path}")

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    entries = _ensure_list(data)
    normalized_entries: list[dict[str, Any]] = []
    changes = 0

    for entry in entries:
        normalized_entry, changed = _normalize_entry(entry)
        normalized_entries.append(normalized_entry)
        if changed:
            changes += 1

    normalized_entries.sort(key=lambda item: item.get("simbolo", ""))
    backup_name = f"assets_catalog_backup_{_dt.date.today().isoformat()}.json"
    backup_path = path.with_name(backup_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(path, backup_path)

    with path.open("w", encoding="utf-8") as fh:
        json.dump(normalized_entries, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")

    return normalized_entries, changes


def main() -> None:
    catalog, changes = sync_catalog()
    total = len(catalog)
    print(
        f"Catálogo sincronizado ({total} activos, {changes} con tipo estandarizado actualizado)."
    )


if __name__ == "__main__":
    main()
