import csv

from shared.cache import VisualCacheRegistry


def test_registry_tracks_hits_and_misses(tmp_path) -> None:
    log_path = tmp_path / "invalidations.csv"
    registry = VisualCacheRegistry(log_path=log_path)

    registry.record("Summary", dataset_hash="alpha", reused=False, signature=("a", "b"))
    registry.record("summary", dataset_hash="alpha", reused=True, signature=("a", "b"))

    snapshot = registry.snapshot()
    entry = snapshot["entries"].get("summary")
    assert entry is not None
    assert entry["misses"] == 1
    assert entry["hits"] == 1

    registry.invalidate("summary", reason="manual_clear")

    assert log_path.exists()
    with log_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert rows[-1]["reason"] == "manual_clear"
    assert rows[-1]["dataset_hash"] == "alpha"
    assert rows[-1]["component"] == "summary"


def test_registry_dataset_invalidation(tmp_path) -> None:
    log_path = tmp_path / "invalidations.csv"
    registry = VisualCacheRegistry(log_path=log_path)

    registry.record("summary", dataset_hash="alpha", reused=True)
    registry.record("table", dataset_hash="beta", reused=True)

    registry.invalidate_dataset("alpha", reason="dataset_hash_changed")

    snapshot = registry.snapshot()
    assert "summary" not in snapshot["entries"]
    table_entry = snapshot["entries"].get("table")
    assert table_entry is not None
    assert table_entry["dataset_hash"] == "beta"

    with log_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert any(row["reason"] == "dataset_hash_changed" and row["dataset_hash"] == "alpha" for row in rows)
