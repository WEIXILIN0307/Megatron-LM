"""Verify the full model's saved global dimensions through the normal DCP reader."""

import json
import sys
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.metadata import TensorStorageMetadata

path = Path(sys.argv[1])
reader = dcp.FileSystemReader(path)
metadata = reader.read_metadata().state_dict_metadata
model = {
    key: item
    for key, item in metadata.items()
    if key.startswith("model.") and isinstance(item, TensorStorageMetadata)
}
count = sum(item.size.numel() for item in model.values())
assert count == 6_919_161_856, count
first = next(iter(model))
state = {first: torch.empty(model[first].size, dtype=model[first].properties.dtype)}
step_keys = [
    key for key in metadata
    if key.startswith("optimizer.param_to_group_meta.") and key.endswith(".step")
]
assert step_keys
for key in step_keys:
    item = metadata[key]
    state[key] = (
        torch.empty(item.size, dtype=item.properties.dtype)
        if isinstance(item, TensorStorageMetadata) else None
    )
planner = dcp.DefaultLoadPlanner(flatten_state_dict=False, allow_partial_load=True)
dcp.load(
    state,
    storage_reader=reader,
    planner=planner,
    no_dist=True,
)
# DCP copies the top-level mapping. Tensor destinations are filled in place,
# but deserialized Python values are assigned into the planner's mapping.
state = planner.state_dict
assert torch.isfinite(state[first]).all()
expected_step = int(path.name.rsplit("_", 1)[1])
steps = {int(state[key]) for key in step_keys}
assert steps == {expected_step}, steps
report = {
    "global_model_parameters": count,
    "model_tensor_count": len(model),
    "total_checkpoint_entries": len(metadata),
    "sample_key": first,
    "sample_shape": list(state[first].shape),
    "sample_finite": True,
    "optimizer_group_step_entries": len(step_keys),
    "optimizer_group_steps": sorted(steps),
}
path.with_suffix(".inspection.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
