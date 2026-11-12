"""Plot recorded kernel intervals from a Nsight Systems SQLite export."""

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def merge(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def intersect(left, right):
    left, right = merge(left), merge(right)
    i = j = 0
    result = []
    while i < len(left) and j < len(right):
        start, end = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if start < end:
            result.append([start, end])
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return result


def load_recording(path):
    if path.suffix == ".json":
        return json.loads(path.read_text())
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    strings = dict(connection.execute("select id,value from StringIds"))
    ranges = [dict(row) for row in connection.execute("select * from NVTX_EVENTS where end is not null")]
    for row in ranges:
        row["name"] = row["text"] or strings.get(row["textId"], "")
    gathers = [row for row in ranges if row["name"].startswith("MFSDP ") and row["name"].endswith(" unshard")]
    reductions = [row for row in ranges if row["name"].startswith("MFSDP ") and row["name"].endswith(" reduce_gradients")]
    runtime = {row["correlationId"]: dict(row) for row in connection.execute("select * from CUPTI_ACTIVITY_KIND_RUNTIME")}
    kernels = []
    for row in connection.execute("select * from CUPTI_ACTIVITY_KIND_KERNEL"):
        name = strings[row["demangledName"]]
        launch = runtime.get(row["correlationId"])
        graph = launch is not None and strings[launch["nameId"]].startswith("cudaGraphLaunch")
        role = None
        owner = None
        if "nccl" not in name.lower():
            role = "graph" if graph else "eager"
        elif not graph and launch is not None:
            owners = gathers if "AllGather" in name else reductions if "ReduceScatter" in name else []
            owners = [item for item in owners if item["start"] <= launch["start"] < item["end"]]
            if owners:
                owner = max(owners, key=lambda item: item["start"])["name"]
                role = "ag" if "AllGather" in name else "expert_rs" if ".mlp.experts " in owner else "ordinary_rs"
        if role is not None:
            interval = [row["start"], row["end"]]
            kernels.append({"role": role, "start_ns": interval[0], "end_ns": interval[1], "stream": row["streamId"], "correlation_id": row["correlationId"], "name": name, "enclosing_nvtx": owner})
    connection.close()
    return {"source_trace": str(path), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "graph_launches": sum(strings[row["nameId"]].startswith("cudaGraphLaunch") for row in runtime.values()), "kernels": kernels}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="Nsight SQLite export or published timeline-data.json")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    recording = load_recording(args.trace)
    kernels = recording["kernels"]
    roles = ["graph", "eager", "ag", "expert_rs", "ordinary_rs"]
    intervals = {key: [] for key in roles}
    streams = {key: set() for key in roles}
    for kernel in kernels:
        intervals[kernel["role"]].append([kernel["start_ns"], kernel["end_ns"]])
        streams[kernel["role"]].add(kernel["stream"])
    ag_overlap = intersect(intervals["graph"], intervals["ag"])
    rs_overlap = intersect(intervals["graph"], intervals["expert_rs"])
    ordinary_overlap = intersect(intervals["graph"], intervals["ordinary_rs"])
    total_ms = lambda items: sum(end - start for start, end in merge(items)) / 1e6
    assert abs(total_ms(ag_overlap) - 26.151144) < 1e-6
    assert abs(total_ms(rs_overlap) - 29.186084) < 1e-6
    assert total_ms(ordinary_overlap) == 0
    origin = min(item["start_ns"] for item in kernels)
    end = max(item["end_ns"] for item in kernels)
    colors = {"graph": "#1675bd", "eager": "#8c98a3", "ag": "#df8a16", "expert_rs": "#8753ad", "ordinary_rs": "#65833a"}
    labels = {"graph": "Graph compute", "eager": "Non-graph compute", "ag": "FSDP AG (meshes aggregated)", "expert_rs": "Expert-DP RS", "ordinary_rs": "Ordinary-DP RS"}
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [1.1, 1, 1]})
    windows = [(0, (end-origin)/1e6)]
    for overlap in [ag_overlap, rs_overlap]:
        start, finish = max(overlap, key=lambda item: item[1]-item[0])
        center = ((start+finish)/2-origin)/1e6
        width = max(4.0, 4*(finish-start)/1e6)
        windows.append((max(0,center-width/2), center+width/2))
    titles = ["A  Sampled optimizer iteration: observed kernels", "B  Zoom: graph compute concurrent with FSDP AG", "C  Zoom: graph compute concurrent with expert-DP RS"]
    for ax, window, title in zip(axes, windows, titles):
        for index, role in enumerate(roles):
            data = [((start-origin)/1e6, (finish-start)/1e6) for start,finish in merge(intervals[role]) if (finish-origin)/1e6 > window[0] and (start-origin)/1e6 < window[1]]
            ax.broken_barh(data, (index-0.32, 0.64), facecolors=colors[role], linewidth=0)
        ax.set_xlim(*window)
        ax.set_ylim(4.6,-0.6)
        ax.set_yticks(range(len(roles)),[labels[role]+"\nCUDA stream "+", ".join(map(str,sorted(streams[role]))) for role in roles],fontsize=9)
        ax.set_title(title,loc="left",fontsize=11,fontweight="bold")
        ax.set_xlabel("Milliseconds from the first recorded kernel in this step",fontsize=9)
        ax.grid(axis="x",alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[["top","right"]].set_visible(False)
    for index, role in enumerate(["ag", "expert_rs"]):
        lo, hi = windows[index+1]
        axes[0].axvspan(lo,hi,color=colors[role],alpha=.25)
    fig.suptitle("Nsight Systems export | rank 0 | 2.09B MoE proxy | EP1 / expert-DP2",fontsize=14,fontweight="bold",x=.54)
    fig.text(.54,.935,"Step 10; 64 graph launches. Recorded durations, not an illustrative schedule.",ha="center",fontsize=10)
    fig.text(.02,.035,"AG/graph intersection: 26.151 ms. Expert-DP RS/graph: 29.186 ms. Ordinary-DP RS/graph: 0 ms.\nKernel concurrency does not establish network progress, hidden critical-path time, or speedup.\nThe export does not pair readiness event records to their producers; dependency arrows are documented separately from source.",fontsize=9)
    fig.subplots_adjust(left=.225,right=.985,top=.89,bottom=.14,hspace=.62)
    fig.savefig(args.output/"nsight-fsdp-graph-timeline.png",dpi=180)
    fig.savefig(args.output/"nsight-fsdp-graph-timeline.svg")
    report = {**recording, "rank": 0, "origin_ns": origin, "stream_ids": {key:sorted(value) for key,value in streams.items()}, "intersection_ms": {"allgather":total_ms(ag_overlap),"expert_dp_reduce_scatter":total_ms(rs_overlap),"ordinary_dp_reduce_scatter":total_ms(ordinary_overlap)}, "windows_ms":windows, "limits":"No invented event producer attribution; no cross-rank clock alignment; no network-progress or critical-path claim"}
    (args.output/"timeline-data.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({key:value for key,value in report.items() if key!="kernels"},indent=2))


if __name__ == "__main__":
    main()
