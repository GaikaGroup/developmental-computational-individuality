import sqlite3

from dci_pilot.detailed_report import _materialize_report_database


def test_report_database_is_runnable_chart_provenance(tmp_path):
    artifact = {
        "manifest": {
            "charts": [{"id": "sample_chart", "dataset": "sample", "sourceId": "raw"}],
            "tables": [],
        },
        "snapshot": {"datasets": {"sample": [{"label": "A", "value": 1.5}]}},
    }
    database = tmp_path / "report.sqlite"

    _materialize_report_database(artifact, database, "2026-08-13T00:00:00Z")

    source = artifact["manifest"]["charts"][0]["source"]
    assert source["query"]["sql"] == 'SELECT * FROM "sample"'
    with sqlite3.connect(database) as connection:
        assert connection.execute(source["query"]["sql"]).fetchall() == [("A", 1.5)]
