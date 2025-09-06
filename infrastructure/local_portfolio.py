import json
from pathlib import Path
from typing import Any, Dict


class LocalPortfolioRepository:
    """Simple JSON-backed storage for portfolio positions.

    The data layout follows the same structure returned by the IOL API:
    {"activos": [ ... ]}
    """

    def __init__(self, path: Path | str = Path(".cache/local_portfolio.json")):
        self.path = Path(path)

    def load(self) -> Dict[str, Any]:
        """Load portfolio from disk returning a dictionary.
        If the file doesn't exist or is invalid, an empty portfolio is returned.
        """
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {"activos": []}

    def save(self, data: Dict[str, Any]) -> None:
        """Persist portfolio data to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # --- Editing helpers -------------------------------------------------
    def add(self, position: Dict[str, Any]) -> None:
        data = self.load()
        data.setdefault("activos", []).append(position)
        self.save(data)

    def update(self, simbolo: str, updates: Dict[str, Any]) -> None:
        data = self.load()
        target = simbolo.upper()
        for it in data.get("activos", []):
            if str(it.get("simbolo", "")).upper() == target:
                it.update(updates)
                break
        self.save(data)

    def remove(self, simbolo: str) -> None:
        data = self.load()
        target = simbolo.upper()
        data["activos"] = [
            it for it in data.get("activos", []) if str(it.get("simbolo", "")).upper() != target
        ]
        self.save(data)