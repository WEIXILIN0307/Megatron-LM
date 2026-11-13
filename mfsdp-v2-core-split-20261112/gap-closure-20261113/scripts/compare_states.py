"""Verify every recorded tensor digest, allowing only the quantified signed-zero witness."""

import gzip
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])


def read(path):
    payload = path.read_bytes() if path.exists() else gzip.decompress(path.with_suffix(path.suffix + ".gz").read_bytes())
    return json.loads(payload)


focus = read(root / "focused-gradient-comparison.json")
assert focus["numerically_equal"] and focus["max_abs_difference"] == 0
assert focus["all_bit_differences_are_signed_zero"] and focus["eager_repeat_bitwise_equal"]
assert focus["canonical_zero_eager_sha256"] == focus["canonical_zero_graph_sha256"]
target = "/step12/gradients_before_optimizer/chunk0/module.module.decoder.layers.13.mlp.experts.linear_fc2.weight3/sha256"


def differences(a, b, prefix=""):
    assert type(a) is type(b), prefix
    if isinstance(a, dict):
        assert a.keys() == b.keys(), prefix
        return [d for key in a for d in differences(a[key], b[key], prefix + "/" + key)]
    if isinstance(a, list):
        assert len(a) == len(b), prefix
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in differences(x, y, prefix + "/" + str(i))]
    return [] if a == b else [{"path": prefix, "eager": a, "graph": b}]


reports = []
for rank in (0, 1):
    a = read(root / f"runs/ep2-state-eager/states-rank-{rank}.json")
    b = read(root / f"runs/ep2-state-graph/states-rank-{rank}.json")
    diff = differences(a["snapshots"], b["snapshots"])
    assert len(a["steps"]) == len(b["steps"]) == 12
    assert [s["loss"] for s in a["steps"]] == [s["loss"] for s in b["steps"]]
    assert all(s["forward_graph_replays"] == (0 if s["step"] < 4 else 32) for s in b["steps"])
    assert all(s["forward_graph_replays"] == 0 for s in a["steps"])
    for item in diff:
        assert rank == 1 and item["path"] == target
        assert item["eager"] == focus["eager_sha256"] and item["graph"] == focus["graph_sha256"]
    assert len(diff) == rank
    for mode in ("eager", "graph"):
        original = a if mode == "eager" else b
        focused = read(root / f"runs/ep2-gradient-{mode}-1/states-rank-{rank}.json")
        assert original["snapshots"] == focused["snapshots"]
        assert original["steps"] == focused["steps"]
    repeated = read(root / f"runs/ep2-gradient-eager-2/states-rank-{rank}.json")
    assert a["snapshots"] == repeated["snapshots"] and a["steps"] == repeated["steps"]
    reports.append({"rank": rank, "all_12_losses_equal": True, "parameters_main_weights_Adam_byte_equal": True,
                    "gradient_numerics_equal": True, "raw_digest_differences": diff,
                    "focused_reruns_reproduce_all_snapshots": True, "eager_repeat_byte_equal": True})
(root / "state-comparison.json").write_text(json.dumps(reports, indent=2) + "\n")
print(json.dumps(reports, indent=2))
