"""Measure one bounded real-text run without profiling or training-state hashing."""

import functools
import json
import os
import runpy
import time
from pathlib import Path

import torch

rank = int(os.environ["RANK"])
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
free_bytes, total_bytes = torch.cuda.mem_get_info()
# Reserve 4 GiB from startup free memory in the PyTorch allocator budget.
# CUDA/NCCL allocations outside that allocator are not covered by this limit.
budget_bytes = free_bytes - 4 * 2**30
if budget_bytes < 16 * 2**30:
    raise RuntimeError("Insufficient free memory for the unchanged MoE proxy")
torch.cuda.set_per_process_memory_fraction(min(1.0, budget_bytes / total_bytes))

import megatron.training.training as training
from megatron.core.transformer.cuda_graphs import TECudaGraphHelper

output = Path(os.environ["RFC_COST_OUTPUT"])
source = Path(os.environ["RFC_COST_SOURCE"])
output.mkdir(parents=True, exist_ok=True)
first = int(os.environ.get("RFC_COST_FIRST", "16"))
last = int(os.environ.get("RFC_COST_LAST", "30"))
report = {
    "rank": rank,
    "variant": os.environ["RFC_COST_VARIANT"],
    "free_bytes_at_start": free_bytes,
    "allocator_budget_bytes": budget_bytes,
    "limits": "Shared GPUs; one bounded window, not evidence of stable speedup. Memory is the process's PyTorch allocator; driver/NCCL allocations are excluded.",
    "capture": [],
    "steps": [],
}
begin = None
original_capture = TECudaGraphHelper.create_cudagraphs
original_step = training.train_step


def memory():
    return {
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "peak_allocated": torch.cuda.max_memory_allocated(),
        "peak_reserved": torch.cuda.max_memory_reserved(),
    }


@functools.wraps(original_capture)
def capture(self):
    torch.cuda.synchronize()
    before = memory()
    start = time.perf_counter()
    result = original_capture(self)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    units = []
    for unit in self.flattened_callables:
        groups = []
        for group in getattr(unit, "parameter_groups", ()):
            buffer = group._unsharded_model_weight
            tensor = buffer.local_buffer
            alias = buffer.is_alias_of(group.model_weight)
            groups.append({
                "elements": tensor.numel(),
                "storage_bytes": tensor.untyped_storage().nbytes(),
                "aliases_model_weight": alias,
                "retained": group._retain_unsharded_storage,
            })
        units.append({"name": getattr(unit, "name", type(unit).__name__), "groups": groups})
    report["capture"].append({"full_setup_seconds": elapsed, "before": before, "after": memory(), "units": units})
    return result


@functools.wraps(original_step)
def step(*args, **kwargs):
    global begin
    number = int(kwargs["iteration"]) + 1
    if number == first:
        torch.distributed.barrier()
        torch.cuda.synchronize()
        report["before_window"] = memory()
        torch.cuda.reset_peak_memory_stats()
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
    # Scalars only, after the timed train_step on the final step.
    report["steps"].append({"step": number, "skipped": int(result[1])})
    return result


TECudaGraphHelper.create_cudagraphs = capture
training.train_step = step
try:
    runpy.run_path(str(source / "pretrain_gpt.py"), run_name="__main__")
finally:
    report["end_memory"] = memory()
    (output / f"cost-rank-{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
