import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

class LocalPortfolioRepository:
    """
    Almacenamiento simple con respaldo JSON para posiciones de cartera.

    El diseño de los datos sigue la misma estructura devuelta por la API de IOL:
    {"activos": [ ... ]}
    """

    def __init__(self, path: Path | str = Path(".cache/local_portfolio.json")):
        self.path = Path(path)

    def load(self) -> Dict[str, Any]:
        """
        Cargar portafolio desde el disco devolviendo un diccionario.
        Si el archivo no existe o no es válido, se devuelve un portafolio vacío.
        """
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("No se pudo leer %s: %s", self.path, e)
            return {"activos": []}

    def save(self, data: Dict[str, Any]) -> None:
        """Conservar los datos de la cartera en el disco."""
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
