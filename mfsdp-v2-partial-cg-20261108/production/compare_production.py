"""Bounded-memory comparison of real DCP model/Adam tensors and training metrics."""

import argparse
import json
import time
import warnings
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.distributed.checkpoint.metadata import TensorStorageMetadata

parser = argparse.ArgumentParser()
parser.add_argument("left")
parser.add_argument("right")
parser.add_argument("--step", type=int, required=True)
parser.add_argument("--batch-mib", type=int, default=1024)
parser.add_argument("--metadata-only", action="store_true")
options = parser.parse_args()
root = Path("/artifacts/production")
report = {"left": options.left, "right": options.right, "step": options.step}


def events(name):
    reader = EventAccumulator(
        str(root / name / "tensorboard"), size_guidance={"scalars": 0}
    )
    reader.Reload()
    tags = [
        tag
        for tag in reader.Tags()["scalars"]
        if tag in ("lm loss", "grad-norm", "params-norm", "num-zeros")
        or "load_balancing_loss" in tag
        or "validation" in tag
    ]
    assert "lm loss" in tags and "grad-norm" in tags and "params-norm" in tags
    # Periodic and final validation can emit the same step. Preserve both events.
    return {tag: [(e.step, e.value) for e in reader.Scalars(tag)] for tag in tags}


start = time.time()
first, second = events(options.left), events(options.right)
assert first.keys() == second.keys(), (first.keys(), second.keys())
report["scalars"] = {}
for tag in first:
    assert [item[0] for item in first[tag]] == [item[0] for item in second[tag]], tag
    assert first[tag] == second[tag], (tag, first[tag], second[tag])
    report["scalars"][tag] = {
        "steps": [item[0] for item in first[tag]],
        "max_abs_difference": 0,
    }
print("All training and validation scalar events are bit exact.", flush=True)

readers = [
    dcp.FileSystemReader(root / name / f"checkpoints/iter_{options.step:07d}")
    for name in (options.left, options.right)
]
metadata = [reader.read_metadata().state_dict_metadata for reader in readers]
keys = [key for key in metadata[0] if key.startswith(("model.", "optimizer."))]
assert keys and set(keys) == {
    key for key in metadata[1] if key.startswith(("model.", "optimizer."))
}
if options.metadata_only:
    keys = [key for key in keys if not isinstance(metadata[0][key], TensorStorageMetadata)]
tensor_keys = [
    key for key in keys if isinstance(metadata[0][key], TensorStorageMetadata)
]
report["tensor_count"] = len(tensor_keys)
report["batch_mib"] = options.batch_mib
report["tensor_elements"] = sum(metadata[0][key].size.numel() for key in tensor_keys)
report["checked_tensors"] = {}
report["checked_non_tensor_entries"] = 0
batch = []
batch_bytes = 0


def check_batch(batch):
    states = []
    for reader, meta in zip(readers, metadata):
        state = {
            key: (
                torch.empty(meta[key].size, dtype=meta[key].properties.dtype)
                if isinstance(meta[key], TensorStorageMetadata)
                else None
            )
            for key in batch
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            planner = dcp.DefaultLoadPlanner(
                flatten_state_dict=False, allow_partial_load=True
            )
            dcp.load(
                state,
                storage_reader=reader,
                planner=planner,
                no_dist=True,
            )
        # Tensors are mutated in place; loaded Python metadata is replaced in
        # the planner mapping because DCP copies the top-level input mapping.
        states.append(planner.state_dict)
    for key in batch:
        a, b = states[0][key], states[1][key]
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b), f"Tensor differs: {key}"
            report["checked_tensors"][key] = {
                "shape": list(a.shape),
                "max_abs_difference": 0,
            }
        else:
            assert a is not None and b is not None, key
            assert a == b, (key, a, b)
            report["checked_non_tensor_entries"] += 1


for key in keys:
    assert type(metadata[0][key]) is type(metadata[1][key]), key
    meta = metadata[0][key]
    if isinstance(meta, TensorStorageMetadata):
        assert meta.size == metadata[1][key].size, key
        assert meta.properties.dtype == metadata[1][key].properties.dtype, key
        size = (
            meta.size.numel()
            * torch.empty((), dtype=meta.properties.dtype).element_size()
        )
    else:
        size = 0
    if batch and batch_bytes + size > options.batch_mib * 1024**2:
        check_batch(batch)
        print(
            f"Checked {len(report['checked_tensors'])}/{len(tensor_keys)} tensors.",
            flush=True,
        )
        batch, batch_bytes = [], 0
    batch.append(key)
    batch_bytes += size
if batch:
    check_batch(batch)
report["passed"] = True
report["wall_seconds"] = time.time() - start
suffix = "-metadata" if options.metadata_only else ""
output = root / f"compare-{options.left}-vs-{options.right}-step{options.step}{suffix}.json"
output.write_text(json.dumps(report, indent=2) + "\n")
print(
    json.dumps(
        {key: value for key, value in report.items() if key != "checked_tensors"},
        indent=2,
    )
)
