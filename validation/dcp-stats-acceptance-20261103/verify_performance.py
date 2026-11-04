"""Check benchmark comparability and metric emission, outside timed training."""

import json
import math
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

base = Path(__file__).resolve().parent
runs = json.loads((base / "current-performance-runs.json").read_text())
assert len(runs) == 9
for run in runs:
    directory = base / run["name"]
    command = (directory / "command.sh").read_text()
    assert "train_measure.py" in command and "DCP_PROFILE_CPU" not in command
    assert not list((directory / "audit").glob("cpu-profile*"))
    assert "--exit-on-missing-checkpoint" in command
    assert "--log-interval 1" in command
    observed = json.loads((directory / "audit/measure-rank1.json").read_text())
    assert len(observed) == 100
    assert all(
        r["tokens"] == 2560 and r["skipped"] == 0 and r["microbatches"] == 3
        for r in observed
    )
    log = (directory / "train.log").read_text()
    events = EventAccumulator(
        str(directory / "tensorboard"), size_guidance={"scalars": 0}
    )
    events.Reload()
    metrics = [key for key in events.Tags()["scalars"] if key.startswith("dcp/")]
    if run["arm"] == "on":
        expected = {
            "dcp/estimated_attention_balance_efficiency": 0.92,
            "dcp/cp_size_1_sample_fraction": 0.875,
            "dcp/cp_size_2_sample_fraction": 0.125,
        }
        assert set(metrics) == set(expected)
        for key, value in expected.items():
            series = events.Scalars(key)
            assert len(series) == 100
            assert [event.step for event in series] == list(range(2, 102))
            assert all(
                math.isclose(event.value, value, rel_tol=1e-6) for event in series
            )
            assert log.count(key + ":") == 100
    else:
        assert not metrics and "dcp/" not in log
report = {
    "runs_checked": 9,
    "checkpoint_restored": True,
    "updates_per_run": 100,
    "warmup_updates_excluded": 10,
    "steady_updates_per_run": 90,
    "tokens_per_update": 2560,
    "microbatches_per_update": 3,
    "once_per_update_stdout_tensorboard": True,
    "primary_runs_without_tensor_hashing_or_cpu_profiling": True,
}
(base / "performance-checks.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
