"""Reproduce the complete tensor signed-zero comparison without GPU or model allocation."""

import gzip
import hashlib
import io
import json
import sys
from pathlib import Path

import torch

root = Path(sys.argv[1])
a, b = [torch.load(io.BytesIO(gzip.decompress((root / "runs" / name / "focused-gradient.pt.gz").read_bytes())),
                   map_location="cpu", weights_only=True)
        for name in ("ep2-gradient-eager-1", "ep2-gradient-graph-1")]
record = json.loads((root / "focused-gradient-comparison.json").read_text())
assert torch.equal(a, b) and torch.isfinite(a).all() and torch.isfinite(b).all()
mask = a.view(torch.int32) != b.view(torch.int32)
assert int(mask.sum()) == record["bitwise_differing_elements"] == 4756
assert bool(((a == 0) & (b == 0))[mask].all())
assert float((a - b).abs().max()) == record["max_abs_difference"] == 0
for value, mode in ((a, "eager"), (b, "graph")):
    assert hashlib.sha256(value.numpy().tobytes()).hexdigest() == record[mode + "_sha256"]
    normalized = torch.where(value == 0, 0., value)
    assert hashlib.sha256(normalized.numpy().tobytes()).hexdigest() == record["canonical_zero_" + mode + "_sha256"]
print("Complete focused gradients are numerically equal; only 4756 signed-zero bits differ.")
