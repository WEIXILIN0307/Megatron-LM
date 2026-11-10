"""Measure GPU kernel overlap, attributing graph kernels by launch correlation."""

import argparse
import bisect
import collections
import gzip
import hashlib
import json
from pathlib import Path


def analyze(path):
    data = path.read_bytes()
    events = json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)["traceEvents"]
    launches = {
        e["args"]["correlation"]: e
        for e in events
        if e.get("name") == "cudaGraphLaunch" and e.get("cat") == "cuda_runtime"
    }
    graph_calls = [
        e for e in events
        if e.get("cat") == "cpu_op" and e.get("name") in {"Graphed", "GraphedBackward"}
    ]
    phases = {}
    for correlation, launch in launches.items():
        parents = [
            e for e in graph_calls
            if e["tid"] == launch["tid"] and e["ts"] <= launch["ts"]
            and e["ts"] + e["dur"] >= launch["ts"] + launch["dur"]
        ]
        phases[correlation] = (
            "backward" if min(parents, key=lambda e: e["dur"])["name"] == "GraphedBackward"
            else "forward"
        ) if parents else "unknown"
    kernels = [e for e in events if e.get("cat") == "kernel"]
    graph = sorted(
        (
            e for e in kernels
            if e.get("args", {}).get("correlation") in launches
            and "nccl" not in e["name"].lower()
        ),
        key=lambda e: e["ts"],
    )
    starts = [e["ts"] for e in graph]
    max_duration = max((e["dur"] for e in graph), default=0)
    groups = collections.defaultdict(
        lambda: {
            "total": 0, "overlapped": 0, "overlap_us": 0,
            "overlap_by_graph_phase": collections.Counter(), "examples": [],
        }
    )
    for event in kernels:
        args = event.get("args", {})
        if args.get("correlation") in launches:
            continue
        is_nccl = "nccl" in event["name"].lower()
        key = (
            args.get("Collective name", "unknown"),
            args.get("Process Group Description", "unknown"),
        ) if is_nccl else ("eager_compute", "all")
        result = groups[key]
        result["total"] += 1
        lo = bisect.bisect_left(starts, event["ts"] - max_duration)
        hi = bisect.bisect_left(starts, event["ts"] + event["dur"])
        matches = [
            g for g in graph[lo:hi]
            if g["ts"] + g["dur"] > event["ts"]
            and g["args"]["stream"] != args["stream"]
            and g["args"]["device"] == args["device"]
        ]
        if not matches:
            continue
        result["overlapped"] += 1
        result["overlap_by_graph_phase"].update(
            {phases[g["args"]["correlation"]] for g in matches}
        )
        spans = sorted(
            (max(event["ts"], g["ts"]), min(event["ts"] + event["dur"], g["ts"] + g["dur"]))
            for g in matches
        )
        merged = []
        for begin, end in spans:
            if merged and begin <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([begin, end])
        result["overlap_us"] += sum(end - begin for begin, end in merged)
        if len(result["examples"]) < 2:
            matched = max(
                matches,
                key=lambda g: min(event["ts"] + event["dur"], g["ts"] + g["dur"])
                - max(event["ts"], g["ts"]),
            )
            result["examples"].append({
                "eager": {k: event[k] for k in ("name", "ts", "dur", "args")},
                "captured": {k: matched[k] for k in ("name", "ts", "dur", "args")},
                "graph_phase": phases[matched["args"]["correlation"]],
            })
    return {
        "trace": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "method": "Actual GPU kernel intervals on distinct streams; graph attribution by cudaGraphLaunch correlation, excluding captured NCCL from compute.",
        "graph_launches": len(launches),
        "graph_launches_by_phase": dict(collections.Counter(phases.values())),
        "captured_compute_kernels": len(graph),
        "graph_streams": dict(collections.Counter(e["args"]["stream"] for e in graph)),
        "groups": [
            {"kind": key[0], "process_group": key[1], **value}
            for key, value in groups.items()
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [analyze(path) for path in args.traces]
    args.output.write_text(json.dumps(reports, indent=2) + "\n")
    for report in reports:
        print(json.dumps({
            "trace": report["trace"],
            "graph_launches": report["graph_launches"],
            "groups": [{k: v for k, v in group.items() if k != "examples"} for group in report["groups"]],
        }))
