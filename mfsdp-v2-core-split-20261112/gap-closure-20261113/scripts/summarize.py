"""Recompute bounded benchmark and event-attribution evidence from per-rank records."""

import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])


def read_run(name):
    directory = root / "runs" / name
    assert json.loads((directory / "exit.json").read_text())["exit_code"] == 0, name
    reports = [json.loads(p.read_text()) for p in sorted(directory.glob("measurement-rank-*.json"))]
    assert reports and all(len(r["steps"]) == 45 and not any(s["skipped"] for s in r["steps"]) for r in reports)
    assert all(r["window"]["first"] == 16 and r["window"]["last"] == 45 for r in reports)
    return {
        "name": name,
        "ms_per_step": max(r["window"]["seconds"] / r["window"]["steps"] * 1000 for r in reports),
        "peak_allocated_GiB": max(r["window"]["memory"]["peak_allocated"] for r in reports) / 2**30,
        "peak_reserved_GiB": max(r["window"]["memory"]["peak_reserved"] for r in reports) / 2**30,
        "capture_setup_s": max((c["setup_seconds"] for r in reports for c in r["capture"]), default=0),
        "control_cpu_enqueue_ms_per_step": max(r.get("barrier_cpu_seconds", 0) / 30 * 1000 for r in reports),
        "losses": [r["losses"] for r in reports],
        "policy": reports[0].get("serialization_policy"),
    }


def compare_topology(prefix):
    modes = {}
    reference = None
    for mode in ("eager", "graph"):
        runs = [read_run(f"{prefix}b{b}-{mode}") for b in (1, 2, 3)]
        for run in runs:
            if reference is None:
                reference = run["losses"]
            assert run.pop("losses") == reference, run["name"]
        modes[mode] = {"runs": runs, "mean_ms_per_step": statistics.mean(r["ms_per_step"] for r in runs)}
    eager, graph = [modes[m]["mean_ms_per_step"] for m in ("eager", "graph")]
    modes.update(step_time_reduction_pct=(1 - graph / eager) * 100,
                 throughput_increase_pct=(eager / graph - 1) * 100,
                 all_45_losses_equal=True)
    return modes, reference


ep1, ep1_losses = compare_topology("steady-")
ep2, _ = compare_topology("ep2-steady-")
serialized = [read_run(n) for n in ("steady-b2-serialized", "steady-b3-serialized", "steady-extra-serialized-v2")]
for run in serialized:
    assert run["policy"] == "graph-boundaries-v2"
    assert run.pop("losses") == ep1_losses
empty = read_run("steady-empty-stream-control")
assert empty.pop("losses") == ep1_losses
controls = {"serialized": serialized,
            "serialized_mean_ms_per_step": statistics.mean(r["ms_per_step"] for r in serialized),
            "empty_stream": empty,
            "all_45_losses_equal": True}
controls["normal_graph_vs_serialized_reduction_pct"] = (1 - ep1["graph"]["mean_ms_per_step"] / controls["serialized_mean_ms_per_step"]) * 100

traces = {}
for name in ("trace-ep2-edp2-events", "trace-ep2-edp2-natural", "trace-ep1-edp3-natural",
             "trace-ep1-edp3-serialized", "trace-ep1-edp3-empty-stream"):
    reports = json.loads((root / "runs" / name / "unit-event-analysis.json").read_text())
    traces[name] = [{k: r[k] for k in ("rank", "native_event_records_available", "event_completion_tracing",
                                     "graph_launches", "native_readiness_chains", "graphs_with_readiness",
                                     "prefetch_graph_intersections", "summary", "graph_internal_communication_kernels")}
                    | {"reduction_and_finish_event_pairs": len(r["reduction_and_finish_event_pairs"])} for r in reports]
report = {"ep1_expert_dp3": ep1, "ep2_expert_dp1": ep2, "serialization_controls": controls, "traces": traces}
(root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"ep1_change_pct": ep1["step_time_reduction_pct"], "ep2_change_pct": ep2["step_time_reduction_pct"],
                  "controls": controls}, indent=2))
