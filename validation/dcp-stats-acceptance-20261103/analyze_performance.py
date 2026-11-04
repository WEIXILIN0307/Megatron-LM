"""Aggregate existing steady-state train timers and returned token counts."""

import json
import re
import statistics
import sys
from pathlib import Path

base = Path(__file__).resolve().parent
prefix = sys.argv[1] if len(sys.argv) > 1 else ""
filename = prefix + "-performance" if prefix else "performance"
runs = json.loads((base / (filename + "-runs.json")).read_text())
rows = []
for run in runs:
    directory = base / run["name"]
    pattern = r"iteration\s+(\d+)/\s*\d+.*?elapsed time per iteration \(ms\): ([0-9.]+)"
    times = [
        (int(i), float(ms))
        for i, ms in re.findall(pattern, (directory / "train.log").read_text())
    ]
    log = (directory / "train.log").read_text()
    assert "successfully loaded checkpoint" in log, run["name"]
    assert "could not find the metadata" not in log, run["name"]
    assert len(times) == 100, (run["name"], len(times))
    assert [i for i, _ in times] == list(range(2, 102))
    # Drop the first ten updates after checkpoint restoration.
    times = times[10:]
    observed = json.loads((directory / "audit/measure-rank1.json").read_text())
    assert len(observed) == len(times) + 10 and all(x["skipped"] == 0 for x in observed)
    for rank in (0, 1):
        assert (
            json.loads((directory / f"audit/measure-rank{rank}.json").read_text())
            == observed
        )
    tokens = {x["iteration"]: x["tokens"] for x in observed}
    assert set(tokens.values()) == {2560.0}
    values = [ms for _, ms in times]
    trim = max(1, int(len(values) * 0.1))
    trimmed = sorted(values)[trim:-trim]
    rows.append(
        {
            **run,
            "steady_updates": len(values),
            "warmup_updates_excluded": 10,
            "mean_ms": statistics.mean(values),
            "median_ms": statistics.median(values),
            "std_ms": statistics.stdev(values),
            "trimmed_mean_ms": statistics.mean(trimmed),
            "effective_tokens_per_second": sum(tokens[i] for i, _ in times)
            / (sum(values) / 1000),
            "microbatch_counts": sorted(set(x["microbatches"] for x in observed)),
        }
    )
aggregate = {}
for arm in ("fixed", "off", "on"):
    group = [r for r in rows if r["arm"] == arm]
    if not group:
        continue
    means = [x["mean_ms"] for x in group]
    aggregate[arm] = {
        "repetitions": len(group),
        "mean_step_ms": statistics.mean(means),
        "between_run_std_ms": statistics.stdev(means) if len(means) > 1 else None,
        "run_mean_ms": means,
        "mean_effective_tokens_per_second": statistics.mean(
            x["effective_tokens_per_second"] for x in group
        ),
        "median_step_ms": statistics.median(x["median_ms"] for x in group),
    }
result = {
    "runs": rows,
    "aggregate": aggregate,
    "source_head": "79bf7b6bcaa95538d2ad7068dc46b7a931fd2c7d",
    "checkpoint_restored_for_all_runs": True,
    "effective_tokens_per_step": 2560,
    "timing": "existing interval-time timer; stdout and TensorBoard enabled; log interval 1; W&B disabled; no tensor hashing or CPU profiling",
    "gpu_indices": [4, 5],
}
if all(k in aggregate for k in ("fixed", "off", "on")):
    result["dcp_on_vs_off_mean_step_change_percent"] = 100 * (
        aggregate["on"]["mean_step_ms"] / aggregate["off"]["mean_step_ms"] - 1
    )
    result["fixed_vs_dcp_off_mean_step_ratio"] = (
        aggregate["fixed"]["mean_step_ms"] / aggregate["off"]["mean_step_ms"]
    )
    result["paired_on_vs_off_step_changes_percent"] = [
        100
        * (
            next(r["mean_ms"] for r in rows if r["arm"] == "on" and r["repeat"] == i)
            / next(r["mean_ms"] for r in rows if r["arm"] == "off" and r["repeat"] == i)
            - 1
        )
        for i in range(
            min(aggregate["on"]["repetitions"], aggregate["off"]["repetitions"])
        )
    ]
(base / (filename + ".json")).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
