import pandas as pd
from unittest.mock import patch

from ui.export import download_csv, PLOTLY_CONFIG


def test_download_csv_uses_streamlit_download_button():
    df = pd.DataFrame({"a": [1]})
    with patch("ui.export.st.download_button") as mock_btn, \
         patch("ui.export.df_to_csv_bytes", return_value=b"csv") as mock_csv:
        download_csv(df, "data.csv", label="Export")
    mock_csv.assert_called_once_with(df)
    mock_btn.assert_called_once_with("Export", b"csv", file_name="data.csv", mime="text/csv")


def test_plotly_config_adds_to_image_button():
    assert "toImage" in PLOTLY_CONFIG.get("modeBarButtonsToAdd", [])
