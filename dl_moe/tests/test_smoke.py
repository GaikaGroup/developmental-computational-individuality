from pathlib import Path

from dl_moe.training import load_config, run_smoke


def test_smoke_pipeline_writes_reproducible_artifact(tmp_path: Path):
    config = load_config("dl_moe/configs/smoke.yaml")
    result = run_smoke(config, tmp_path, "dl_moe", "interleaved", seed=0)
    assert result["smoke_only"] is True
    assert result["task_counts"] == {"A": 10, "B": 10}
    assert list(tmp_path.glob("*.json"))
