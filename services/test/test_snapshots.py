from __future__ import annotations

import pytest

from services import snapshots


def test_json_snapshot_backend_persists_and_compares(tmp_path):
    storage_file = tmp_path / "snapshots.json"
    snapshots.configure_storage(backend="json", path=storage_file)

    payload_a = {
        "totals": {"total_value": 100.0, "total_cost": 80.0, "total_pl": 20.0},
        "history": [],
    }
    payload_b = {
        "totals": {"total_value": 90.0, "total_cost": 75.0, "total_pl": 15.0},
        "history": [],
    }

    first = snapshots.save_snapshot("portfolio", payload_a, {"dataset_key": "foo"})
    second = snapshots.save_snapshot("portfolio", payload_b, {"dataset_key": "bar"})

    records = snapshots.list_snapshots("portfolio", limit=2, order="desc")
    assert len(records) == 2
    assert records[0]["id"] == second["id"]

    comparison = snapshots.compare_snapshots(first["id"], second["id"])
    assert comparison is not None
    assert comparison["delta"]["total_value"] == pytest.approx(10.0)
    assert comparison["totals_b"]["total_cost"] == pytest.approx(75.0)

    snapshots.configure_storage(backend="null")


def test_sqlite_snapshot_backend_persists_and_compares(tmp_path):
    db_path = tmp_path / "snapshots.db"
    snapshots.configure_storage(backend="sqlite", path=db_path)

    payload_a = {
        "totals": {"total_value": 150.0, "total_cost": 110.0, "total_pl": 40.0},
        "history": [],
    }
    payload_b = {
        "totals": {"total_value": 120.0, "total_cost": 95.0, "total_pl": 25.0},
        "history": [],
    }

    try:
        first = snapshots.save_snapshot("portfolio", payload_a, {"dataset_key": "foo"})
        second = snapshots.save_snapshot("portfolio", payload_b, {"dataset_key": "bar"})

        records = snapshots.list_snapshots("portfolio", limit=2, order="desc")
        assert len(records) == 2
        assert records[0]["id"] == second["id"]

        # Reconfigure to exercise reopening the database connection.
        snapshots.configure_storage(backend="sqlite", path=db_path)
        reopened_records = snapshots.list_snapshots("portfolio", limit=2, order="desc")
        assert [record["id"] for record in reopened_records] == [second["id"], first["id"]]

        comparison = snapshots.compare_snapshots(first["id"], second["id"])
        assert comparison is not None
        assert comparison["delta"]["total_value"] == pytest.approx(30.0)
        assert comparison["totals_b"]["total_cost"] == pytest.approx(95.0)
    finally:
        snapshots.configure_storage(backend="null")
