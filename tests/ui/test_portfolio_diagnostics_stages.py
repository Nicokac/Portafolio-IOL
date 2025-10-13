from types import SimpleNamespace

import pandas as pd

from ui.panels import diagnostics as panel


class FakeStreamlit:
    def __init__(self):
        self.session_state = {}
        self.headers = []
        self.subheaders = []
        self.dataframes = []
        self.captions = []
        self.downloads = []

    def header(self, label: str) -> None:
        self.headers.append(label)

    def subheader(self, label: str) -> None:
        self.subheaders.append(label)

    def dataframe(self, frame: pd.DataFrame, **kwargs) -> None:
        self.dataframes.append((frame, kwargs))

    def caption(self, text: str) -> None:
        self.captions.append(text)

    def columns(self, count: int):
        return [SimpleNamespace(metric=lambda *a, **k: None) for _ in range(count)]

    def download_button(self, *args, **kwargs):
        self.downloads.append((args, kwargs))

    def divider(self):
        return None


def test_diagnostics_panel_renders_stage_timings(monkeypatch):
    fake_st = FakeStreamlit()
    monkeypatch.setattr(panel, "st", fake_st)
    monkeypatch.setattr(panel, "get_recent_metrics", lambda: [])
    monkeypatch.setattr(
        panel,
        "get_cache_stats",
        lambda: SimpleNamespace(hits=1, misses=0, hit_ratio=1.0, ttl_hours=0.5, last_updated="2024-01-01"),
    )
    monkeypatch.setattr(panel, "export_metrics_csv", lambda: b"csv")
    monkeypatch.setattr(panel, "safe_page_link", lambda *a, **k: None)

    fake_st.session_state["portfolio_stage_timings"] = {"render_summary": 12.3, "render_table": 4.5}

    panel.render_diagnostics_panel()

    assert fake_st.subheaders and "Última renderización" in fake_st.subheaders[0]
    stage_frame = fake_st.dataframes[0][0]
    assert list(stage_frame.columns) == ["Subetapa", "Duración (ms)"]
    assert set(stage_frame["Subetapa"]) == {
        "Render Summary",
        "Render Table",
    }
