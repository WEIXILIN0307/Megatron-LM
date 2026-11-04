"""Verify real DP2xCP2 training from raw schedules using independent arithmetic."""

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

BASE = Path(__file__).resolve().parent
BALANCE = "dcp/estimated_attention_balance_efficiency"


def read(name, rank):
    return [
        json.loads(line)
        for line in (BASE / name / "audit" / f"rank{rank}.jsonl")
        .read_text()
        .splitlines()
    ]


def normalized(stats):
    return {
        **{k: v for k, v in stats.items() if k != "rank_work"},
        "cp_sample_counts": {
            str(k): v for k, v in stats["cp_sample_counts"].items() if v
        },
    }


def independent(schedule, lengths):
    lengths = dict(lengths)
    work = []
    counts = Counter()
    for microbatch in schedule:
        assert len(microbatch) == 4
        replicas = Counter(sid for rank in microbatch for sid in set(rank))
        row = [
            math.fsum(int(lengths[sid]) ** 2 / replicas[sid] for sid in rank)
            for rank in microbatch
        ]
        assert math.isclose(
            math.fsum(row),
            math.fsum(int(lengths[sid]) ** 2 for sid in replicas),
            rel_tol=1e-12,
        )
        work.append(row)
        counts.update(str(c) for c in replicas.values())
    return {
        "rank_work": work,
        "balance_numerator": math.fsum(map(math.fsum, work)),
        "balance_denominator": 4 * math.fsum(max(row) for row in work),
        "cp_sample_counts": dict(counts),
    }


def empty():
    return {"balance_numerator": 0, "balance_denominator": 0, "cp_sample_counts": {}}


def add(window, stats):
    for key in ("balance_numerator", "balance_denominator"):
        window[key] += stats[key]
    window["cp_sample_counts"] = dict(
        Counter(window["cp_sample_counts"]) + Counter(stats["cp_sample_counts"])
    )


def metrics(stats):
    n = sum(stats["cp_sample_counts"].values())
    return {
        BALANCE: stats["balance_numerator"] / stats["balance_denominator"],
        **{
            f"dcp/cp_size_{c}_sample_fraction": value / n
            for c, value in stats["cp_sample_counts"].items()
            if value
        },
    }


expected_events = {}
step_summaries = []
window_summaries = []
for rank in range(4):
    off, on = read("dp2-off", rank), read("dp2-on", rank)
    source_off = next(r["hashes"] for r in off if r["kind"] == "source")
    source_on = next(r["hashes"] for r in on if r["kind"] == "source")
    assert source_off == source_on
    assert all(
        hashlib.sha256(Path(path).read_bytes()).hexdigest() == value
        for path, value in source_on.items()
    )
    a = [r for r in off if r["kind"] == "schedule"]
    b = [r for r in on if r["kind"] == "schedule"]
    assert a == b
    steps_off = [r for r in off if r["kind"] == "step"]
    steps_on = [r for r in on if r["kind"] == "step"]
    assert len(steps_on) == len(steps_off) == 9
    assert [r["iteration"] for r in steps_on] == list(range(1, 10))
    assert [{k: v for k, v in r.items() if k != "stats"} for r in steps_on] == [
        {k: v for k, v in r.items() if k != "stats"} for r in steps_off
    ]
    assert all(r["stats"] is None and r["skipped"] == 0 for r in steps_off)
    train_schedules = [r for r in b if r["phase"] == "train"]
    assert len(train_schedules) == 9
    expected = [independent(r["schedule"], r["lengths"]) for r in train_schedules]
    assert len({r["balance_numerator"] for r in expected}) >= 3
    assert len({tuple(sorted(r["cp_sample_counts"].items())) for r in expected}) >= 3
    assert all(sum(r["cp_sample_counts"].values()) == 8 for r in expected)
    for calculation, schedule_row, step in zip(expected, train_schedules, steps_on):
        assert calculation == schedule_row["stats"]
        assert normalized(calculation) == normalized(step["stats"])
        assert step["skipped"] == 0
    windows = [r for r in on if r["kind"] == "window"]
    assert len(windows) == 9
    accumulated = empty()
    for calculated, window in zip(expected, windows):
        iteration = window["iteration"]
        add(accumulated, calculated)
        assert normalized(window["before"]) == normalized(accumulated)
        if iteration % 3 == 0:
            if rank == 3:
                expected_events[iteration] = metrics(accumulated)
                window_summaries.append(
                    {
                        "iteration": iteration,
                        **accumulated,
                        "metrics": metrics(accumulated),
                    }
                )
            accumulated = empty()
        assert normalized(window["after"]) == normalized(accumulated)
    # The unlogged last iteration remains pending; completed windows were reset.
    assert normalized(accumulated) == normalized(expected[-1])
    assert all(r["before"] == r["after"] for r in on if r["kind"] == "eval_boundary")
    assert sum(r["kind"] == "eval_boundary" for r in on) >= 3
    assert any(r["phase"] == "eval" for r in b if r["kind"] == "schedule")
    assert all(
        r["before"] is None and r["after"] is None for r in off if r["kind"] == "window"
    )
    if rank == 0:
        step_summaries = [normalized(r) for r in expected]
    else:
        assert [normalized(r) for r in expected] == step_summaries

for name in ("dp2-off", "dp2-on"):
    path = BASE / name
    log = (path / "train.log").read_text()
    assert "successfully loaded checkpoint" in log
    assert "could not find the metadata" not in log
    event = EventAccumulator(str(path / "tensorboard"), size_guidance={"scalars": 0})
    event.Reload()
    keys = [k for k in event.Tags()["scalars"] if k.startswith("dcp/")]
    if name.endswith("off"):
        assert not keys and "dcp/" not in log
    else:
        wanted = {key for row in expected_events.values() for key in row}
        assert set(keys) == wanted
        for key in wanted:
            observed = event.Scalars(key)
            expected = [
                (step, row[key]) for step, row in expected_events.items() if key in row
            ]
            assert len(observed) == len(expected)
            for actual, (step, value) in zip(observed, expected):
                assert actual.step == step and math.isclose(
                    actual.value, value, rel_tol=1e-6
                )
            stdout = re.findall(re.escape(key) + r": ([0-9.]+)", log)
            # Existing logging also prints the first resumed iteration (2) to stdout.
            assert len(stdout) == len(expected) + 1
            first = metrics(step_summaries[0])[key]
            values = [first] + [v for _, v in expected]
            assert all(
                math.isclose(float(a), b, abs_tol=5.01e-7)
                for a, b in zip(stdout, values)
            )

report = {
    "source_head": "79bf7b6bcaa95538d2ad7068dc46b7a931fd2c7d",
    "topology": {"world": 4, "TP": 1, "PP": 1, "DP": 2, "CP": 2, "dp_cp_domain": 4},
    "updates_per_arm": 9,
    "bitwise_off_on": [
        "schedules",
        "scheduled_microbatch_counts",
        "losses",
        "gradients",
        "updated_parameters",
    ],
    "log_interval": 3,
    "tensorboard_event_iterations": sorted(expected_events),
    "stdout_event_iterations": [2, *sorted(expected_events)],
    "evaluation_excluded": True,
    "per_microbatch_conservation": True,
    "global_summaries_identical_across_ranks": True,
    "completed_windows_reset": True,
    "last_partial_window_not_emitted": True,
    "steps": step_summaries,
    "windows": window_summaries,
}
(BASE / "dp2-correctness.json").write_text(json.dumps(report, indent=2) + "\n")
print(
    json.dumps(
        {k: v for k, v in report.items() if k not in ("steps", "windows")}, indent=2
    )
)
print("window calculations:", json.dumps(window_summaries, indent=2))
