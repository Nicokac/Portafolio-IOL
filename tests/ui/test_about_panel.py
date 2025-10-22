from __future__ import annotations

from typing import Any

import pytest

from ui.panels import about


class _FakeStreamlit:
    def __init__(self) -> None:
        self.headers: list[str] = []
        self.captions: list[str] = []
        self.expanders: list[str] = []
        self.json_payloads: list[dict[str, Any]] = []

    class _Expander:
        def __init__(self, parent: "_FakeStreamlit", label: str) -> None:
            self.parent = parent
            self.label = label

        def __enter__(self) -> "_FakeStreamlit._Expander":
            self.parent.expanders.append(self.label)
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    def header(self, message: str) -> None:
        self.headers.append(message)

    def caption(self, message: str) -> None:
        self.captions.append(message)

    def json(self, payload: dict[str, Any]) -> None:
        self.json_payloads.append(payload)

    def expander(self, label: str) -> "_FakeStreamlit._Expander":
        return self._Expander(self, label)


def test_render_about_panel_shows_update_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_st = _FakeStreamlit()
    history = [
        {
            "timestamp": f"2024-05-{day:02d}",
            "event": "update",
            "version": f"0.5.{day}",
            "status": "done",
        }
        for day in range(1, 13)
    ]

    monkeypatch.setattr(about, "st", fake_st)
    monkeypatch.setattr(about, "get_update_history", lambda: history)
    monkeypatch.setattr(about.platform, "system", lambda: "TestOS")
    monkeypatch.setattr(about.platform, "release", lambda: "1.0")
    monkeypatch.setattr(about.platform, "python_version", lambda: "3.11.0")
    monkeypatch.setattr(about.tempfile, "gettempdir", lambda: "/tmp/test")
    monkeypatch.setattr(about.os, "getcwd", lambda: "/app")
    monkeypatch.setattr(about.os, "environ", {"FOO": "BAR"})
    monkeypatch.setattr(about, "__version__", "0.6.0", raising=False)

    about.render_about_panel()

    assert fake_st.headers == ["ℹ️ Acerca de Portafolio-IOL"]
    assert any("Versión: v0.6.0" in caption for caption in fake_st.captions)
    event_captions = [c for c in fake_st.captions if c.startswith("🕒 ")]
    assert len(event_captions) == 10
    assert fake_st.expanders == [
        "📜 Últimos eventos de actualización",
        "🧠 Información del entorno",
    ]
    assert fake_st.json_payloads == [{"cwd": "/app", "environment": {"FOO": "BAR"}}]
