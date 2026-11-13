"""Measure matched training windows; optional serialization is a measurement control."""

import functools
import json
import os
import runpy
import time
from pathlib import Path

import torch

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
free_bytes, total_bytes = torch.cuda.mem_get_info()
budget_bytes = free_bytes - 4 * 2**30
if budget_bytes < 18 * 2**30:
    raise RuntimeError("Insufficient capacity for the unchanged 2.09B MoE proxy")
torch.cuda.set_per_process_memory_fraction(min(1.0, budget_bytes / total_bytes))

import megatron.training.training as training
from megatron.core.transformer.cuda_graphs import TECudaGraphHelper

if os.environ.get("PROOF_CAPTURE_PROVENANCE") == "1":
    import trace_units

output = Path(os.environ["PROOF_OUTPUT"])
source = Path(os.environ["PROOF_SOURCE"])
rank = int(os.environ["RANK"])
first = int(os.environ.get("PROOF_FIRST", "16"))
last = int(os.environ.get("PROOF_LAST", "45"))
report = {
    "rank": rank,
    "mode": os.environ["PROOF_MODE"],
    "free_bytes_at_start": free_bytes,
    "allocator_budget_bytes": budget_bytes,
    "capture": [],
    "steps": [],
    "serialized_gather_boundaries": 0,
    "serialized_reduction_boundaries": 0,
    "measure_active": False,
}
begin = None
losses = []


def memory():
    return {
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "peak_allocated": torch.cuda.max_memory_allocated(),
        "peak_reserved": torch.cuda.max_memory_reserved(),
    }


if report["mode"] in ("serialized", "empty-stream"):
    from serialize_fsdp import install

    install(report, empty_streams=report["mode"] == "empty-stream")


original_capture = TECudaGraphHelper.create_cudagraphs
original_step = training.train_step


@functools.wraps(original_capture)
def capture(self):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = original_capture(self)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    units = []
    for unit in self.flattened_callables:
        groups = []
        for group in getattr(unit, "parameter_groups", ()):
            buffer = group._unsharded_model_weight
            groups.append({
                "storage_bytes": buffer.local_buffer.untyped_storage().nbytes(),
                "aliases_model_weight": buffer.is_alias_of(group.model_weight),
                "retained": group._retain_unsharded_storage,
            })
        units.append({"name": getattr(unit, "name", type(unit).__name__), "groups": groups})
    report["capture"].append({"setup_seconds": elapsed, "memory": memory(), "units": units})
    return result


@functools.wraps(original_step)
def step(*args, **kwargs):
    global begin
    number = int(kwargs["iteration"]) + 1
    if number == first:
        torch.distributed.barrier()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        report["measure_active"] = True
        begin = time.perf_counter()
    result = original_step(*args, **kwargs)
    if number == last:
        torch.cuda.synchronize()
        report["window"] = {
            "first": first,
            "last": last,
            "steps": last - first + 1,
            "seconds": time.perf_counter() - begin,
            "memory": memory(),
        }
        report["measure_active"] = False
    report["steps"].append({"step": number, "skipped": int(result[1])})
    # Keep only scalar tensors; read them after the timed window, without per-step sync.
    losses.append({k: v.detach() for k, v in result[0].items()
                   if torch.is_tensor(v) and v.numel() == 1})
    return result


TECudaGraphHelper.create_cudagraphs = capture
training.train_step = step
try:
    runpy.run_path(str(source / "pretrain_gpt.py"), run_name="__main__")
finally:
    report["end_memory"] = memory()
    report["losses"] = [{k: v.item() for k, v in row.items()} for row in losses]
    if os.environ.get("PROOF_CAPTURE_PROVENANCE") == "1":
        report["capture_collectives"] = trace_units.capture_collectives
    (output / f"measurement-rank-{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
