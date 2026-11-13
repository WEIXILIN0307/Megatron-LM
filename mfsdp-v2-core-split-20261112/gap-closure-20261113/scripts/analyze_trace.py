"""Attribute actual collectives and match readiness event generations to graph calls."""

import bisect
import gzip
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path


def merge(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def intersection(first, second):
    a, b = merge(first), merge(second)
    i = j = 0
    result = []
    while i < len(a) and j < len(b):
        start, end = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if start < end:
            result.append([start, end])
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return result


def duration_ms(intervals):
    return sum(end - start for start, end in merge(intervals)) / 1e6


def event_key(row):
    # These Nsight tables use different device-index spaces on nonzero ranks.
    # PID + CUDA context identifies the device without equating those ordinals.
    return tuple(row[key] for key in ("globalPid", "contextId", "eventId", "eventSyncId"))


def analyze(directory, rank):
    ownership = json.loads((directory / f"ownership-rank-{rank}.json").read_text())
    metadata = {r["id"]: r for r in ownership["records"]}
    groups = ownership["groups"]
    trace = directory / f"trace-rank-{rank}.sqlite"
    if trace.exists():
        connection = sqlite3.connect(trace.as_uri() + "?mode=ro", uri=True)
    else:
        connection = sqlite3.connect(":memory:")
        connection.deserialize(gzip.decompress(trace.with_suffix(".sqlite.gz").read_bytes()))
    connection.row_factory = sqlite3.Row
    tables = {r[0] for r in connection.execute("select name from sqlite_master where type='table'")}
    strings = dict(connection.execute("select id,value from StringIds"))
    ranges = [dict(r) for r in connection.execute("select * from NVTX_EVENTS where end is not null")]
    by_kind_thread = defaultdict(list)
    ranges_by_id = {}
    native_backward = defaultdict(list)
    for index, row in enumerate(ranges):
        row["name"] = row["text"] or strings.get(row["textId"], "")
        row["index"] = index
        if row["name"].startswith("proof|"):
            fields = row["name"].split("|")
            row["kind"] = fields[1]
            row["proof_id"] = int(fields[2].split("=", 1)[1])
            by_kind_thread[(row["kind"], row["globalTid"])].append(row)
            ranges_by_id[row["proof_id"]] = row
        elif row["name"].startswith("MFSDP ") and row["name"].endswith(" backward"):
            row["unit"] = row["name"][6:-9]
            native_backward[row["globalTid"]].append(row)
    starts = {}
    for key, rows in by_kind_thread.items():
        rows.sort(key=lambda r: r["start"])
        starts[key] = [r["start"] for r in rows]

    def scope(api, kind):
        key = (kind, api["globalTid"])
        rows = by_kind_thread.get(key, ())
        index = bisect.bisect_right(starts.get(key, ()), api["start"]) - 1
        return rows[index] if index >= 0 and api["start"] < rows[index]["end"] else None

    runtime = {r["correlationId"]: dict(r) for r in connection.execute("select * from CUPTI_ACTIVITY_KIND_RUNTIME")}
    for row in runtime.values():
        row["name"] = strings[row["nameId"]]
    events = ([dict(r) for r in connection.execute("select * from CUPTI_ACTIVITY_KIND_CUDA_EVENT")]
              if "CUPTI_ACTIVITY_KIND_CUDA_EVENT" in tables else [])
    events_by_key = {event_key(r): r for r in events}
    event_by_correlation = {r["correlationId"]: r for r in events}
    syncs = [dict(r) for r in connection.execute("select * from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION")]
    sync_by_correlation = defaultdict(list)
    for row in syncs:
        sync_by_correlation[row["correlationId"]].append(row)

    graph_units = {r["unit"] for r in metadata.values() if r["kind"] == "graph_call"}
    graphs = {}
    for api in runtime.values():
        if not api["name"].startswith("cudaGraphLaunch"):
            continue
        replay_scope = scope(api, "replay")
        assert replay_scope is not None
        call_scope = scope(api, "graph_call")
        if call_scope is not None:
            unit = metadata[call_scope["proof_id"]]["unit"]
            phase = "forward"
            invocation = call_scope["proof_id"]
            boundary = call_scope["start"]
        else:
            candidates = [r for r in native_backward[api["globalTid"]]
                          if r["unit"] in graph_units and r["start"] <= api["start"] < r["end"]]
            assert candidates, api
            parent = max(candidates, key=lambda r: r["start"])
            unit, phase, invocation, boundary = parent["unit"], "backward", parent["index"], parent["start"]
        graphs[api["correlationId"]] = {
            "correlation": api["correlationId"], "unit": unit, "phase": phase,
            "invocation": invocation, "boundary": boundary, "api_start": api["start"],
            "api_end": api["end"], "globalTid": api["globalTid"],
            "stream_native": metadata[replay_scope["proof_id"]]["stream_native"],
            "kernels": [], "communication_kernels": [],
        }

    operations = []
    unrelated_nccl = []
    for raw in connection.execute("select * from CUPTI_ACTIVITY_KIND_KERNEL"):
        kernel = dict(raw)
        api = runtime.get(kernel["correlationId"])
        if api is None:
            continue
        name = strings[kernel["demangledName"]]
        thin = {"start": kernel["start"], "end": kernel["end"], "stream": kernel["streamId"],
                "name": name, "correlation": kernel["correlationId"]}
        if kernel["correlationId"] in graphs:
            if "nccl" in name.lower():
                graphs[kernel["correlationId"]]["communication_kernels"].append(thin)
            else:
                graphs[kernel["correlationId"]]["kernels"].append(thin)
        elif "nccl" in name.lower():
            owner = next((scope(api, kind) for kind in ("AG", "RS", "A2A") if scope(api, kind)), None)
            if owner is None:
                unrelated_nccl.append(thin)
                continue
            row = metadata[owner["proof_id"]]
            if row["kind"] == "A2A":
                category = "ep_" + row["stage"]
            else:
                dimensions = groups[row["group"]]["mesh_dim_names"]
                assert dimensions in (["dp"], ["expert_dp"]), dimensions
                category = ("expert_dp_" if "expert_dp" in dimensions else "ordinary_dp_") + row["kind"]
            operations.append({**thin, "category": category, "proof_id": row["id"],
                               "unit": row.get("unit"), "group": row.get("group")})
    assert not any("AllGather" in k["name"] or "ReduceScatter" in k["name"] for k in unrelated_nccl)
    assert all(graph["kernels"] for graph in graphs.values())
    all_compute = [[k["start"], k["end"]] for g in graphs.values() for k in g["kernels"]]
    summary = {}
    for category in sorted({r["category"] for r in operations}):
        selected = [r for r in operations if r["category"] == category]
        intervals = [[r["start"], r["end"]] for r in selected]
        summary[category] = {"kernels": len(selected), "kernel_union_ms": duration_ms(intervals),
                             "graph_intersection_ms": duration_ms(intersection(intervals, all_compute))}

    record_apis = defaultdict(list)
    wait_apis = defaultdict(list)
    for api in runtime.values():
        if "EventRecord" in api["name"]:
            parent = scope(api, "unshard")
            if parent:
                record_apis[parent["proof_id"]].append(api)
        if "StreamWaitEvent" in api["name"]:
            parent = scope(api, "wait")
            if parent:
                wait_apis[parent["proof_id"]].append(api)
    readiness = {}
    for identifier, apis in record_apis.items():
        row = metadata[identifier]
        if not any(r["producer"] == identifier for r in row["readiness"]):
            continue
        api = max(apis, key=lambda r: r["start"])
        readiness[identifier] = {"metadata": row, "api": api,
                                  "native_event": event_by_correlation.get(api["correlationId"])}
    chains = []
    stream_mapping = {}
    for row in metadata.values():
        if row["kind"] != "wait":
            continue
        expected = row["readiness"]
        apis = sorted(wait_apis[row["id"]], key=lambda r: r["start"])
        assert len(apis) == len(expected), (row, apis)
        for api, expected_event in zip(apis, expected):
            producer = readiness[expected_event["producer"]]
            waits = [r for r in sync_by_correlation[api["correlationId"]] if r["syncType"] == 2]
            if events:
                assert len(waits) == 1, (row, waits)
                native = events_by_key[event_key(waits[0])]
                producer_api = runtime[native["correlationId"]]
                producer_scope = scope(producer_api, "unshard")
                assert producer_scope is not None and producer_scope["proof_id"] == expected_event["producer"]
                producer = {**producer, "native_event": native, "api": producer_api}
                matched = waits
            else:
                native = None
                matched = []
            if waits:
                actual = waits[0]["streamId"]
                assert stream_mapping.setdefault(row["stream_native"], actual) == actual
            chain = {"wait_id": row["id"], "group": row["group"], "unit": row["unit"],
                     "phase": row["phase"], "handle": expected_event["handle"],
                     "producer_id": expected_event["producer"],
                     "record_correlation": producer["api"]["correlationId"],
                     "wait_correlation": api["correlationId"], "native_event": native,
                     "native_wait": matched[0] if matched else None, "graphs": []}
            ag_kernels = [k for k in operations if k["category"].endswith("_AG")
                          and metadata[k["proof_id"]]["producer"] == expected_event["producer"]]
            chain["AG_kernels_on_record_stream"] = (all(k["stream"] == native["streamId"]
                and runtime[k["correlation"]]["start"] < producer["api"]["start"] for k in ag_kernels)
                if native is not None else None)
            chain["AG_correlations"] = [k["correlation"] for k in ag_kernels]
            wait_range = ranges_by_id[row["id"]]
            for graph in graphs.values():
                if (graph["unit"] == row["unit"] and graph["phase"] == row["phase"]
                        and graph["globalTid"] == api["globalTid"]
                        and graph["boundary"] <= wait_range["start"] < graph["api_start"]
                        and graph["stream_native"] == row["stream_native"]):
                    chain["graphs"].append(graph["correlation"])
            chains.append(chain)

    witnesses = []
    for operation in operations:
        if not operation["category"].endswith("_AG"):
            continue
        ag = metadata[operation["proof_id"]]
        unshard = metadata[ag["producer"]]
        cursor = unshard["cursor"]
        if cursor is None or cursor["unit"] == operation["unit"]:
            continue
        order = ownership["contexts"][groups[operation["group"]]["context"]]["order"]
        successor = {"unit": operation["unit"], "phase": unshard["phase"]}
        assert successor in order[cursor["position"] + 1:], (cursor, successor)
        api = runtime[operation["correlation"]]
        for graph in graphs.values():
            if graph["unit"] != cursor["unit"] or graph["phase"] != cursor["phase"]:
                continue
            if not (graph["boundary"] <= api["start"] < graph["api_start"]):
                continue
            own_chains = [c for c in chains if graph["correlation"] in c["graphs"]]
            assert own_chains
            assert all(c["producer_id"] != ag["producer"] for c in own_chains)
            overlap = intersection([[operation["start"], operation["end"]]],
                                   [[k["start"], k["end"]] for k in graph["kernels"]])
            witnesses.append({"caller_unit": graph["unit"], "phase": graph["phase"],
                              "graph_correlation": graph["correlation"], "target_unit": operation["unit"],
                              "AG_correlation": operation["correlation"], "target_group": operation["group"],
                              "prefetch_producer_id": ag["producer"],
                              "current_readiness_wait_ids": [c["wait_id"] for c in own_chains],
                              "intersection_ms": duration_ms(overlap)})
    generic_pairs = []
    for wait in syncs:
        producer = events_by_key.get(event_key(wait))
        api = runtime.get(wait["correlationId"])
        if producer is None or api is None:
            continue
        for kind in ("pack_reduce", "finish_grad_sync"):
            owner = scope(api, kind)
            if owner:
                generic_pairs.append({"scope_id": owner["proof_id"], "scope": kind,
                                      "producer": producer, "consumer": wait})
    requested = json.loads((directory / "exit.json").read_text())["options"]["cuda_event_trace"]
    return {"rank": rank, "native_event_records_available": bool(events),
            "event_completion_tracing": requested == "true", "summary": summary,
            "graph_launches": len(graphs), "readiness_chains": chains,
            "native_readiness_chains": sum(c["native_wait"] is not None for c in chains),
            "graphs_with_readiness": len({g for c in chains for g in c["graphs"]}),
            "prefetch_witnesses": witnesses,
            "prefetch_graph_intersections": sum(w["intersection_ms"] > 0 for w in witnesses),
            "reduction_and_finish_event_pairs": generic_pairs, "stream_native_to_cupti": stream_mapping,
            "groups": groups, "graphs": list(graphs.values()), "operations": operations,
            "graph_internal_communication_kernels": sum(len(g["communication_kernels"]) for g in graphs.values()),
            "other_nccl": unrelated_nccl,
            "limits": "Kernel intersection is concurrency. Device-side event tracing can perturb scheduling; compare with event tracing disabled and measure performance separately."}


directory = Path(sys.argv[1]).resolve()
reports = []
for path in sorted(directory.glob("ownership-rank-*.json")):
    rank = int(path.stem.rsplit("-", 1)[1])
    report = analyze(directory, rank)
    reports.append(report)
    print(json.dumps({key: report[key] for key in ("rank", "native_event_records_available", "event_completion_tracing", "graph_launches",
          "native_readiness_chains", "graphs_with_readiness", "prefetch_graph_intersections", "summary")}))
(directory / "unit-event-analysis.json").write_text(json.dumps(reports, indent=2) + "\n")
