"""Plot actual EP2/expert-DP2 kernels and concrete prefetch/reduction examples."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(sys.argv[1] if len(sys.argv) > 1 else "/data01/LWX/mfsdp-gap-closure-20261113")
report = json.loads((root / "runs/trace-ep2-edp2-natural/unit-event-analysis.json").read_text())[0]
graphs = {g["correlation"]: g for g in report["graphs"]}
operations = report["operations"]
base = min(k["start"] for g in graphs.values() for k in g["kernels"])
palette = {"graph": "#2364aa", "graph_metadata_AG": "#9c755f", "ordinary_dp_AG": "#f28e2b",
           "expert_dp_AG": "#e15759", "ordinary_dp_RS": "#59a14f", "expert_dp_RS": "#b07aa1",
           "EP_A2A": "#76b7b2"}


def segment(axis, kernels, row, color, origin=base):
    for k in kernels:
        axis.broken_barh([((k["start"] - origin) / 1e6, (k["end"] - k["start"]) / 1e6)],
                        (row - 0.28, 0.56), facecolors=color, linewidth=0)


fig, axes = plt.subplots(3, 1, figsize=(15, 10), gridspec_kw={"height_ratios": [2.5, 1.3, 1.3]})
ax = axes[0]
compute = [k for g in graphs.values() for k in g["kernels"]]
metadata = [k for g in graphs.values() for k in g["communication_kernels"]]
segment(ax, compute, 6, palette["graph"])
segment(ax, metadata, 5, palette["graph_metadata_AG"])
for row, category in [(4, "ordinary_dp_AG"), (3, "expert_dp_AG"), (2, "ordinary_dp_RS"), (1, "expert_dp_RS")]:
    segment(ax, [k for k in operations if k["category"] == category], row, palette[category])
segment(ax, [k for k in operations if k["category"].startswith("ep_")], 0, palette["EP_A2A"])
ax.set_yticks(range(7), ["EP dispatch/combine A2A", "Expert-DP RS", "Ordinary-DP RS",
                       "Expert-DP AG", "Ordinary-DP AG", "EP counts AG (inside graph)", "Graph compute"])
ax.set_title("A. Recorded kernels: rank 0, four A6000s, EP=2 / expert-DP=2", loc="left", weight="bold")
ax.set_xlabel("Time from first graph-compute kernel (ms)")

witness = next(w for w in report["prefetch_witnesses"] if w["intersection_ms"] > 0)
graph = graphs[witness["graph_correlation"]]
ag = next(k for k in operations if k["correlation"] == witness["AG_correlation"])
chain = next(c for c in report["readiness_chains"] if c["wait_id"] == witness["current_readiness_wait_ids"][0])
origin = min(ag["start"], min(k["start"] for k in graph["kernels"])) - 200000
end = max(ag["end"], max(k["end"] for k in graph["kernels"])) + 200000
ax = axes[1]
segment(ax, [ag], 1, palette["expert_dp_AG"], origin)
segment(ax, graph["kernels"], 0, palette["graph"], origin)
ax.set_xlim(0, (end - origin) / 1e6)
ax.set_yticks([0, 1], ["Graph: layer 0 forward", "Prefetch: layer 0 experts"])
event = chain["native_event"]
wait = chain["native_wait"]
ax.set_title("B. Actual successor prefetch overlaps independent graph compute", loc="left", weight="bold")
ax.set_xlabel("Local time (ms)")
ax.text(0.01, -0.65,
        f"Current-layer readiness: event {event['eventId']} / generation {event['eventSyncId']}, "
        f"record API {chain['record_correlation']} on stream {event['streamId']} → "
        f"wait API {chain['wait_correlation']} on stream {wait['streamId']}.\n"
        f"Graph launch {graph['correlation']} does not wait on successor producer "
        f"{witness['prefetch_producer_id']}; measured kernel intersection = {witness['intersection_ms']:.3f} ms.",
        transform=ax.transAxes, fontsize=9, va="top")

pair = None
for graph in graphs.values():
    if graph["phase"] != "backward":
        continue
    for operation in operations:
        if operation["category"] != "expert_dp_RS" or not operation["unit"].startswith(graph["unit"] + "."):
            continue
        if any(max(k["start"], operation["start"]) < min(k["end"], operation["end"]) for k in graph["kernels"]):
            pair = graph, operation
            break
    if pair:
        break
assert pair is not None
graph, rs = pair
origin = min(rs["start"], min(k["start"] for k in graph["kernels"])) - 200000
end = max(rs["end"], max(k["end"] for k in graph["kernels"])) + 200000
ax = axes[2]
segment(ax, [rs], 1, palette["expert_dp_RS"], origin)
segment(ax, graph["kernels"], 0, palette["graph"], origin)
ax.set_xlim(0, (end - origin) / 1e6)
layer = graph["unit"].split("layers.")[1]
ax.set_yticks([0, 1], [f"Graph: layer {layer} backward", f"RS owner: layer {layer} experts"])
ax.set_title("C. Ready expert gradients reduce alongside attention/router backward compute", loc="left", weight="bold")
ax.set_xlabel("Local time (ms)")
ax.text(0.01, -0.55, f"Graph launch {graph['correlation']}; expert-DP RS launch {rs['correlation']}. "
        "AG/RS owners come from actual target groups and DeviceMesh dimensions.",
        transform=ax.transAxes, fontsize=9, va="top")
for ax in axes:
    ax.grid(axis="x", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("FSDP v2 partial CUDA Graph: observed execution and event identity", fontsize=15, weight="bold")
fig.text(0.01, 0.005, "CUDA event-completion tracing disabled. Shared fourth GPU; this is concurrency evidence, "
         "not a timing benchmark. Event labels identify host submission APIs, not invented GPU record timestamps.", fontsize=9)
fig.subplots_adjust(left=0.20, right=0.98, top=0.92, bottom=0.15, hspace=0.90)
fig.savefig(root / "fsdp-unit-event-timeline.png", dpi=160)
fig.savefig(root / "fsdp-unit-event-timeline.svg")
