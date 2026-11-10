"""Show measured kernel intersections and one EP all-to-all counterexample."""

import gzip
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
report = json.loads((root / "overlap-filtered.json").read_text())[0]
report["trace"] = str(root / "rank-0-filtered.json.gz")
raw = Path(report["trace"]).read_bytes()
assert hashlib.sha256(raw).hexdigest() == report["sha256"]
events = json.loads(gzip.decompress(raw))["traceEvents"]
launches = {e["args"]["correlation"]: e for e in events
            if e.get("name") == "cudaGraphLaunch" and e.get("cat") == "cuda_runtime"}
kernels = [e for e in events if e.get("cat") == "kernel"]
captured = [e for e in kernels if e.get("args", {}).get("correlation") in launches
            and "nccl" not in e["name"].lower()]
pairs = []
for kind, title in [("_allgather_base", "Forward: eager FSDP all-gather + captured GEMM"),
                    ("_reduce_scatter_base", "Backward: eager FSDP reduce-scatter + captured GEMM")]:
    group = next(g for g in report["groups"] if g["kind"] == kind
                 and g["process_group"] == "DATA_PARALLEL_GROUP_WITH_CP")
    pairs.append({"title": title, "kind": kind, **group["examples"][1]})

alltoall = [e for e in kernels if e.get("args", {}).get("Collective name") == "all_to_allv"
            and e["args"].get("Process Group Description") == "EXPERT_MODEL_PARALLEL_GROUP"]
assert len(alltoall) == 192
choices = []
for eager in alltoall:
    same_gpu = [g for g in captured if g["args"]["device"] == eager["args"]["device"]
                and g["args"]["context"] == eager["args"]["context"]]
    assert not any(g["ts"] < eager["ts"] + eager["dur"]
                   and g["ts"] + g["dur"] > eager["ts"] for g in same_gpu)
    before = [g for g in same_gpu if g["ts"] + g["dur"] <= eager["ts"]]
    if before:
        graph = max(before, key=lambda g: g["ts"] + g["dur"])
        gap = eager["ts"] - graph["ts"] - graph["dur"]
        if gap >= 3:
            choices.append((eager["ts"] + eager["dur"] - graph["ts"], eager, graph))
_, eager, graph = min(choices, key=lambda item: item[0])
pairs.append({"title": "EP all-to-all: no intersection (one measured example)",
              "kind": "all_to_allv", "eager": eager, "captured": graph})

fig, axes = plt.subplots(3, 1, figsize=(11.5, 8), layout="constrained")
for ax, pair in zip(axes, pairs):
    eager, graph = pair["eager"], pair["captured"]
    assert eager["args"]["correlation"] not in launches
    assert graph["args"]["correlation"] in launches
    assert eager["args"]["device"] == graph["args"]["device"]
    assert eager["args"]["context"] == graph["args"]["context"]
    if pair["kind"] != "all_to_allv":
        assert eager["args"]["stream"] != graph["args"]["stream"]
    origin = min(eager["ts"], graph["ts"])
    begin = max(eager["ts"], graph["ts"])
    end = min(eager["ts"] + eager["dur"], graph["ts"] + graph["dur"])
    overlap = max(0, end - begin)
    for event, row, color in [(eager, 1, "#d9872a"), (graph, 0, "#2867a6")]:
        ax.broken_barh([(event["ts"] - origin, event["dur"])], (row - .2, .4), color=color)
    if overlap > 0:
        ax.axvspan(begin - origin, end - origin, color="#319951", alpha=.2)
    span = max(eager["ts"] + eager["dur"], graph["ts"] + graph["dur"]) - origin
    ax.set_xlim(-span * .025, span * 1.04)
    ax.set_ylim(-.55, 1.65)
    ax.set_yticks([1, 0], [f"Eager NCCL / stream {eager['args']['stream']}",
                          f"Graph kernel / stream {graph['args']['stream']}"])
    ax.set_title(pair["title"], fontsize=11, loc="left")
    ax.set_xlabel("Relative GPU time (microseconds)")
    ax.text(.99, .9, f"Intersection: {overlap:.3f} us", transform=ax.transAxes,
            ha="right", color="#207040" if overlap else "#555555")
    ax.grid(axis="x", alpha=.18)
    ax.spines[["top", "right", "left"]].set_visible(False)
    pair["origin_us"] = origin
    pair["overlap_us"] = overlap
    pair["graph_launch"] = launches[graph["args"]["correlation"]]

fig.suptitle("Measured GPU kernel overlap in full OLMoE training\n"
             "4 x RTX A6000 | 6.919B parameters | S=1024 | EP=2 | rank 0", fontsize=13)
fig.supxlabel("Each panel has its own time origin. Shading marks actual kernel intersection.\n"
              "EP all-to-all: 0 / 192 intersections in this trace. No speedup claim.", fontsize=9)
fig.savefig(root / "overlap-observation.svg")
fig.savefig(root / "overlap-observation.png", dpi=160)
(root / "overlap-observation.json").write_text(json.dumps({
    "trace": report["trace"], "trace_sha256": report["sha256"], "examples": pairs,
}, indent=2) + "\n")
for pair in pairs:
    a, b = pair["eager"], pair["captured"]
    print(json.dumps({"kind": pair["kind"], "overlap_us": pair["overlap_us"],
        "eager_stream": a["args"]["stream"], "graph_stream": b["args"]["stream"],
        "eager_interval_us": [a["ts"] - pair["origin_us"], a["ts"] + a["dur"] - pair["origin_us"]],
        "graph_interval_us": [b["ts"] - pair["origin_us"], b["ts"] + b["dur"] - pair["origin_us"]]}))
