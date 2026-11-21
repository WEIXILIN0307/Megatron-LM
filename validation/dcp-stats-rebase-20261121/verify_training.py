"""Verify actual TP2xCP2 replay from raw schedules, FLOPs inputs and events."""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

BASE = Path(__file__).resolve().parent
KEY = "dcp/estimated_attention_balance_efficiency"


def read(arm, rank):
    return [
        json.loads(line)
        for line in (BASE / f"tp-{arm}/audit/rank{rank}.jsonl").read_text().splitlines()
    ]


def summary(stats):
    return {k: v for k, v in stats.items() if k != "rank_work"}


def calculate(row):
    lengths = dict(row["lengths"])
    work, counts = [], Counter()
    for microbatch in row["schedule"]:
        assert len(microbatch) == 2  # TP replicas are not part of DPxCP.
        replicas = Counter(sid for ids in microbatch for sid in set(ids))
        ranks = [
            math.fsum(lengths[sid] ** 2 / replicas[sid] for sid in ids)
            for ids in microbatch
        ]
        assert math.fsum(ranks) == sum(lengths[sid] ** 2 for sid in replicas)
        work.append(ranks)
        counts.update(str(n) for n in replicas.values())
    return {
        "rank_work": work,
        "balance_numerator": math.fsum(map(math.fsum, work)),
        "balance_denominator": 2 * math.fsum(max(r) for r in work),
        "cp_sample_counts": dict(counts),
    }


def empty():
    return {"balance_numerator": 0, "balance_denominator": 0, "cp_sample_counts": {}}


canonical = None
windows = None
for rank in range(4):
    off, on = read("off", rank), read("on", rank)
    source_a = next(r["hashes"] for r in off if r["kind"] == "source")
    source_b = next(r["hashes"] for r in on if r["kind"] == "source")
    assert source_a == source_b
    assert all(
        hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
        for path, digest in source_a.items()
    )
    a = [r for r in off if r["kind"] == "step"]
    b = [r for r in on if r["kind"] == "step"]
    assert len(a) == len(b) == 5
    assert [r["iteration"] for r in b] == [1, 2, 3, 4, 5]
    assert [{k: v for k, v in r.items() if k != "stats"} for r in a] == [
        {k: v for k, v in r.items() if k != "stats"} for r in b
    ]
    assert all(r["stats"] is None for r in a)
    assert len({r["params"] for r in b}) == 5
    assert all(r["grads"] != hashlib.sha256(b"").hexdigest() for r in b)
    assert all(
        r["skipped"] == 0
        and r["microbatches"] == 3
        and r["legacy_flops"] == [2560, 1507328]
        for r in b
    )
    flops_off = [r for r in off if r["kind"] == "flops_input"]
    flops_on = [r for r in on if r["kind"] == "flops_input"]
    assert flops_off == flops_on and len(flops_on) == 5
    assert all(
        r["phase"] == "train" and r["values"] == [2560, 1507328] and r["kwargs"] == {}
        for r in flops_on
    )
    schedules_off = [r for r in off if r["kind"] == "schedule"]
    schedules_on = [r for r in on if r["kind"] == "schedule"]
    assert schedules_off == schedules_on
    if rank % 2 == 0:
        train = [r for r in schedules_on if r["phase"] == "train"]
        assert len(train) == 5
        expected = [calculate(r) for r in train]
        assert all(r["stats"] == value for r, value in zip(train, expected))
        assert all(step["stats"] == value for step, value in zip(b, expected))
        assert all(value["cp_sample_counts"] == {"1": 7, "2": 1} for value in expected)
        assert any(r["phase"] == "eval" for r in schedules_on)
        if canonical is None:
            canonical = [summary(value) for value in expected]
        else:
            assert canonical == [summary(value) for value in expected]
    else:
        assert not schedules_on
        assert all("rank_work" not in step["stats"] for step in b)
    assert canonical == [summary(step["stats"]) for step in b]
    assert not [r for r in off if r["kind"] == "production_stats"]
    production = [r for r in on if r["kind"] == "production_stats"]
    assert len(production) == (5 if rank % 2 == 0 else 0)
    assert all(r["phase"] == "train" for r in production)
    accumulated = empty()
    rank_windows = []
    for value, row in zip(canonical, [r for r in on if r["kind"] == "window"]):
        for key in ("balance_numerator", "balance_denominator"):
            accumulated[key] += value[key]
        accumulated["cp_sample_counts"] = dict(
            Counter(accumulated["cp_sample_counts"])
            + Counter(value["cp_sample_counts"])
        )
        assert row["before"] == accumulated
        if row["iteration"] % 2 == 0:
            rank_windows.append({"iteration": row["iteration"], **accumulated})
            accumulated = empty()
        assert row["after"] == accumulated
    assert accumulated == empty()
    assert [r["iteration"] for r in rank_windows] == [2, 4, 6]
    if windows is None:
        windows = rank_windows
    else:
        assert windows == rank_windows
    evaluations = [r for r in on if r["kind"] == "eval_boundary"]
    assert len(evaluations) >= 2 and all(r["before"] == r["after"] for r in evaluations)
    assert all(
        r["before"] is None and r["after"] is None for r in off if r["kind"] == "window"
    )

expected_metrics = {
    KEY: 0.92,
    "dcp/cp_size_1_sample_fraction": 0.875,
    "dcp/cp_size_2_sample_fraction": 0.125,
}
for arm in ("off", "on"):
    path = BASE / f"tp-{arm}"
    log = (path / "train.log").read_text()
    assert "successfully loaded checkpoint" in log
    assert "could not find the metadata" not in log
    events = EventAccumulator(str(path / "tensorboard"), size_guidance={"scalars": 0})
    events.Reload()
    keys = [key for key in events.Tags()["scalars"] if key.startswith("dcp/")]
    if arm == "off":
        assert not keys and "dcp/" not in log
    else:
        assert set(keys) == set(expected_metrics)
        for key, value in expected_metrics.items():
            observed = events.Scalars(key)
            assert [r.step for r in observed] == [2, 4, 6]
            assert all(math.isclose(r.value, value, rel_tol=1e-6) for r in observed)
            assert log.count(key + ":") == 3
report = {
    "tested_runtime_head": "e3311dba708a30e8d8418e491d4e15003750bbd6",
    "upstream_base": "6559f64c27f2c547a3b48e18bd6f7326d980df94",
    "topology": {"world": 4, "TP": 2, "CP": 2, "PP": 1, "DP": 1, "dp_cp_domain": 2},
    "updates_per_arm": 5,
    "bitwise_off_on": [
        "final_schedules",
        "scheduled_microbatches",
        "legacy_FLOPs_return_fields",
        "legacy_training_FLOPs_inputs",
        "losses",
        "gradient_hashes",
        "updated_parameter_hashes",
    ],
    "metrics_only_when_enabled": True,
    "production_statistics_only_on_TP_sources": True,
    "evaluation_excluded": True,
    "per_microbatch_conservation": True,
    "all_TP_peers_receive_same_summary": True,
    "log_interval": 2,
    "stdout_tensorboard_iterations": [2, 4, 6],
    "metrics": expected_metrics,
    "windows": windows,
    "benchmark_rerun": False,
}
(BASE / "correctness.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
