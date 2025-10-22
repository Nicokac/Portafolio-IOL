# ui\palette.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Palette:
    """Color palette with accessibility in mind."""

    bg: str
    plot_bg: str
    grid: str
    text: str
    positive: str
    negative: str
    accent: str
    categories: Dict[str, str]
    highlight_bg: str
    highlight_text: str


# Okabe & Ito colorblind-friendly palette
OI_BLUE = "#0072B2"
OI_ORANGE = "#E69F00"
OI_SKY = "#56B4E9"
OI_GREEN = "#009E73"
OI_YELLOW = "#F0E442"
OI_VERMILION = "#D55E00"
OI_PURPLE = "#CC79A7"

PALETTES = {
    "light": Palette(
        bg="#FFFFFF",
        plot_bg="#FFFFFF",
        grid="rgba(0,0,0,0.08)",
        text="#262626",
        positive=OI_GREEN,
        negative=OI_VERMILION,
        accent=OI_BLUE,
        categories={
            "CEDEAR": OI_BLUE,
            "Bono": OI_GREEN,
            "Acción": OI_VERMILION,
            "ETF": OI_PURPLE,
            "FCI": OI_ORANGE,
            "Letra": OI_SKY,
            "Otro": OI_YELLOW,
        },
        highlight_bg=OI_BLUE,
        highlight_text="#FFFFFF",
    ),
    "dark": Palette(
        bg="#0e1117",
        plot_bg="#0e1117",
        grid="rgba(255,255,255,0.08)",
        text="#e5e5e5",
        positive=OI_GREEN,
        negative=OI_VERMILION,
        accent=OI_SKY,
        categories={
            "CEDEAR": OI_SKY,
            "Bono": OI_GREEN,
            "Acción": OI_VERMILION,
            "ETF": OI_PURPLE,
            "FCI": OI_ORANGE,
            "Letra": OI_BLUE,
            "Otro": OI_YELLOW,
        },
        highlight_bg=OI_SKY,
        highlight_text="#0e1117",
    ),
}


def get_palette(theme: str = "light") -> Palette:
    return PALETTES.get(theme, PALETTES["light"])


def get_active_palette() -> Palette:
    from .ui_settings import get_settings

    return get_palette(get_settings().theme)
