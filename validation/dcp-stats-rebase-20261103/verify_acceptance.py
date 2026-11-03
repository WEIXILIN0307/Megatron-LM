"""Check the retained real-training audits and existing TensorBoard scalars."""

import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

BASE = Path(__file__).resolve().parent
KEY = "dcp/estimated_attention_balance_efficiency"


def records(directory, rank):
    return [
        json.loads(line)
        for line in (directory / "audit" / f"rank{rank}.jsonl").read_text().splitlines()
    ]


def summary(stats):
    return {k: v for k, v in stats.items() if k != "rank_work"}


def metrics(stats):
    total = sum(stats["cp_sample_counts"].values())
    return {
        KEY: stats["balance_numerator"] / stats["balance_denominator"],
        **{
            f"dcp/cp_size_{c}_sample_fraction": n / total
            for c, n in stats["cp_sample_counts"].items()
            if n
        },
    }


report = {}
for name, world, tp in [("2", 2, 1), ("tp", 4, 2), ("pp", 4, 1)]:
    off, on = BASE / f"final-{name}-off", BASE / f"final-{name}-on"
    expected_metrics = []
    schedule_rows = []
    for rank in range(world):
        a, b = records(off, rank), records(on, rank)
        source_a = [x for x in a if x["kind"] == "source"][0]["hashes"]
        source_b = [x for x in b if x["kind"] == "source"][0]["hashes"]
        assert source_a == source_b
        assert all(
            hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
            for p, h in source_a.items()
        )
        schedules_a = [x for x in a if x["kind"] == "schedule"]
        schedules_b = [x for x in b if x["kind"] == "schedule"]
        assert schedules_a == schedules_b, (name, rank, "schedule mismatch")
        steps_a = [x for x in a if x["kind"] == "step"]
        steps_b = [x for x in b if x["kind"] == "step"]
        assert len(steps_a) == len(steps_b) == 5
        for x, y in zip(steps_a, steps_b):
            assert {k: v for k, v in x.items() if k != "stats"} == {
                k: v for k, v in y.items() if k != "stats"
            }, (name, rank, x["iteration"], "step mismatch")
            assert not y["skipped"] and y["stats"]
            assert x["stats"] is None
            assert y["grads"] != hashlib.sha256().hexdigest()
            assert all(math.isfinite(v) for v in y["losses"].values())
        assert steps_b[0]["params"] != steps_b[-1]["params"]
        if rank % tp == 0:
            train_schedules = [x for x in schedules_b if x["phase"] == "train"]
            assert len(train_schedules) == 5
            assert any(x["phase"] == "eval" for x in schedules_b)
            for schedule, step in zip(train_schedules, steps_b):
                lengths = dict(schedule["lengths"])
                assert len(schedule["schedule"]) == step["microbatches"]
                assert summary(schedule["stats"]) == summary(step["stats"])
                assert sum(schedule["stats"]["cp_sample_counts"].values()) == 8
                for batch, row in zip(
                    schedule["schedule"], schedule["stats"]["rank_work"]
                ):
                    assert len(row) == 2
                    ids = set(i for rank_ids in batch for i in rank_ids)
                    assert math.isclose(sum(row), sum(lengths[i] ** 2 for i in ids))
                if name == "pp":
                    assert (
                        schedule["unaligned_microbatches"] == 3
                        and len(schedule["schedule"]) == 4
                    )
                    assert schedule["stats"]["cp_sample_counts"] == {"1": 3, "2": 5}
                else:
                    assert schedule["stats"]["cp_sample_counts"] == {"1": 7, "2": 1}
                assert schedule["stats"]["balance_denominator"] == 2 * sum(
                    max(row) for row in schedule["stats"]["rank_work"]
                )
            schedule_rows = train_schedules
        elif tp > 1:
            assert not schedules_b
            assert all("rank_work" not in x["stats"] for x in steps_b)
        if rank == 0:
            expected_metrics = [
                (x["iteration"] + 1, metrics(x["stats"])) for x in steps_b
            ]
        else:
            assert [
                (x["iteration"] + 1, metrics(x["stats"])) for x in steps_b
            ] == expected_metrics
    # Exactly one stdout record and one TensorBoard record per trained iteration.
    assert KEY not in (off / "train.log").read_text()
    lines = []
    for line in (on / "train.log").read_text().splitlines():
        if KEY in line:
            iteration = int(re.search(r"iteration\s+(\d+)/", line).group(1))
            values = {
                k: float(v) for k, v in re.findall(r"(dcp/[\w_]+): ([0-9.]+)", line)
            }
            lines.append((iteration, values))
    assert len(lines) == 5
    for (step, expected), (logged_step, values) in zip(expected_metrics, lines):
        assert step == logged_step
        assert values.keys() == expected.keys()
        assert all(abs(expected[k] - values[k]) < 1e-6 for k in expected)
    events = EventAccumulator(
        str(on / "tensorboard"), size_guidance={"scalars": 0}
    ).Reload()
    for key in expected_metrics[0][1]:
        scalars = events.Scalars(key)
        assert len(scalars) == 5
        for event, (step, expected) in zip(scalars, expected_metrics):
            assert event.step == step and abs(event.value - expected[key]) < 1e-7
    report[name] = dict(
        ranks=world,
        dp_cp_ranks=2,
        updates=5,
        bitwise_equal=[
            "schedule",
            "microbatch_count",
            "loss",
            "gradient",
            "updated_parameter",
        ],
        metrics=expected_metrics[0][1],
        vpp_microbatches_before=3 if name == "pp" else None,
        vpp_microbatches_after=4 if name == "pp" else None,
        training_log_records=5,
        tensorboard_records_per_metric=5,
        evaluation_excluded=True,
        production_source_hashes=source_a,
    )
(BASE / "correctness.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
