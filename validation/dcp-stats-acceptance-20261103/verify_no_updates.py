"""Check the actual loop-without-updates audit and training metric outputs."""

import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

base = Path(__file__).resolve().parent
path = base / "no-updates"
rows = [
    json.loads((path / "audit" / f"rank{rank}.json").read_text()) for rank in (0, 1)
]
assert rows[0] == rows[1]
for row in rows:
    assert row["data_hook_iterations"] == [0, 1, 2]
    assert (
        row["empty_submissions"] == row["empty_log_windows"] == row["iterations"] == 3
    )
    assert row["parameters_unchanged"] is True
assert "dcp/" not in (path / "train.log").read_text()
events = EventAccumulator(str(path / "tensorboard"))
events.Reload()
assert not any(key.startswith("dcp/") for key in events.Tags()["scalars"])
report = {
    "ranks": 2,
    "iterations": 3,
    "data_hook_called_each_iteration": True,
    "train_step_calls": 0,
    "schedule_statistics_calls": 0,
    "parameters_unchanged": True,
    "dcp_stdout_tensorboard_metrics": 0,
}
(base / "no-updates-check.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
