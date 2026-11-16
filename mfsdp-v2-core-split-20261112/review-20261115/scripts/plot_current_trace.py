"""Plot current-head kernels and native event identities, using actual owner labels."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root, output = map(Path, sys.argv[1:3])
report = json.loads((root / "unit-event-analysis.json").read_text())[0]
owners = {r["id"]: r for r in json.loads((root / "ownership-rank-0.json").read_text())["records"]}
graphs = {r["correlation"]: r for r in report["graphs"]}
operations = report["operations"]
compute = [k for g in graphs.values() for k in g["kernels"]]
origin = min(k["start"] for k in compute)
colors = {"compute": "#2364aa", "metadata": "#9c755f", "ordinary_dp_AG": "#f28e2b",
          "expert_dp_AG": "#e15759", "ordinary_dp_RS": "#59a14f", "expert_dp_RS": "#b07aa1",
          "a2a": "#76b7b2"}


def draw(ax, kernels, row, color, base=origin):
    for k in kernels:
        ax.broken_barh([((k["start"] - base) / 1e6, (k["end"] - k["start"]) / 1e6)],
                      (row - .27, .54), facecolor=color, linewidth=0)


def short(unit):
    return unit.replace("module.decoder.", "")


fig, axes = plt.subplots(3, 1, figsize=(15, 10), gridspec_kw={"height_ratios": [2.2, 1.2, 1.2]})
ax = axes[0]
draw(ax, compute, 6, colors["compute"])
draw(ax, [k for g in graphs.values() for k in g["communication_kernels"]], 5, colors["metadata"])
for row, category in [(4, "ordinary_dp_AG"), (3, "expert_dp_AG"), (2, "ordinary_dp_RS"), (1, "expert_dp_RS")]:
    draw(ax, [k for k in operations if k["category"] == category], row, colors[category])
draw(ax, [k for k in operations if k["category"].startswith("ep_")], 0, colors["a2a"])
ax.set_yticks(range(7), ["EP dispatch/combine A2A", "Expert-DP RS", "Ordinary-DP RS",
                       "Expert-DP AG", "Ordinary-DP AG", "EP counts AG (inside graph)", "Graph compute"])
ax.set_title("A. Current implementation: four A6000s, EP=2 / expert-DP=2, rank 0", loc="left", weight="bold")
ax.set_xlabel("Time from first graph-compute kernel (ms)")

witness = next(r for r in report["prefetch_witnesses"] if r["intersection_ms"] > 0)
graph = graphs[witness["graph_correlation"]]
ag = next(k for k in operations if k["correlation"] == witness["AG_correlation"])
chain = next(r for r in report["readiness_chains"] if r["wait_id"] == witness["current_readiness_wait_ids"][0])
start = min(ag["start"], min(k["start"] for k in graph["kernels"])) - 100000
ax = axes[1]
draw(ax, [ag], 1, colors[ag["category"]], start)
draw(ax, graph["kernels"], 0, colors["compute"], start)
ax.set_yticks([0, 1], ["Graph: " + short(graph["unit"]), "Prefetch: " + short(ag["unit"])])
ax.set_title("B. Actual successor gather overlaps independent graph compute", loc="left", weight="bold")
ax.set_xlabel("Local time (ms)")
event, wait = chain["native_event"], chain["native_wait"]
ax.text(.01, -.65, f"Own readiness: event {event['eventId']} / generation {event['eventSyncId']}, "
        f"stream {event['streamId']} → {wait['streamId']}; record/wait APIs "
        f"{chain['record_correlation']}/{chain['wait_correlation']}.\n"
        f"Graph {graph['correlation']}; successor AG {ag['correlation']}; kernel intersection "
        f"{witness['intersection_ms']:.3f} ms. Successor is a distinct readiness dependency.",
        transform=ax.transAxes, va="top", fontsize=9)

pair = next((g, op) for g in graphs.values() if g["phase"] == "backward"
            for op in operations if op["category"] == "expert_dp_RS"
            and op["unit"].startswith(g["unit"] + ".")
            and any(max(k["start"], op["start"]) < min(k["end"], op["end"]) for k in g["kernels"]))
graph, rs = pair
start = min(rs["start"], min(k["start"] for k in graph["kernels"])) - 100000
ax = axes[2]
draw(ax, [rs], 1, colors[rs["category"]], start)
draw(ax, graph["kernels"], 0, colors["compute"], start)
ax.set_yticks([0, 1], ["Graph backward: " + short(graph["unit"]), "RS owner: " + short(rs["unit"])])
ax.set_title("C. Expert gradients reduce alongside independent parent backward compute", loc="left", weight="bold")
ax.set_xlabel("Local time (ms)")
pack_id = max(r["id"] for r in owners.values() if r["kind"] == "pack_reduce"
              and r["group"] == rs["group"] and r["id"] < rs["proof_id"])
edge = next(r for r in report["reduction_and_finish_event_pairs"] if r["scope_id"] == pack_id
            and r["consumer"]["streamId"] == rs["stream"])
ev, consumer = edge["producer"], edge["consumer"]
ax.text(.01, -.55, f"Pack/reduction event {ev['eventId']} / generation {ev['eventSyncId']}: "
        f"stream {ev['streamId']} → {consumer['streamId']}. Graph {graph['correlation']}; RS {rs['correlation']}.",
        transform=ax.transAxes, va="top", fontsize=9)
for ax in axes:
    ax.grid(axis="x", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("FSDP v2 partial graph: current-head execution and event dependencies", fontsize=15, weight="bold")
fig.text(.01, .012, "Focused GPT MoE probe: 2 layers, hidden512, 4 experts/top2, seq16, GA2. "
         "Event-completion tracing disabled. Kernel intervals establish concurrency, not speedup or network progress.", fontsize=9)
fig.subplots_adjust(left=.23, right=.98, top=.92, bottom=.16, hspace=.95)
fig.savefig(output, dpi=160)
