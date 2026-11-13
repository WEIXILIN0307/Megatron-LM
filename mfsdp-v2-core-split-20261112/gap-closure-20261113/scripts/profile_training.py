"""Capture one optimizer step with ownership labels and CUDA event completion tracing."""

import functools
import json
import os
import runpy
from pathlib import Path

import torch

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
free_bytes, total_bytes = torch.cuda.mem_get_info()
budget = free_bytes - 4 * 2**30
torch.cuda.set_per_process_memory_fraction(min(1.0, budget / total_bytes))

import megatron.training.training as training
import trace_units

output = Path(os.environ["PROOF_OUTPUT"])
source = Path(os.environ["PROOF_SOURCE"])
rank = int(os.environ["RANK"])
original_step = training.train_step
steps = []
serialization = {"serialized_gather_boundaries": 0, "serialized_reduction_boundaries": 0}
if os.environ["PROOF_MODE"] in ("serialized", "empty-stream"):
    from serialize_fsdp import install

    install(serialization, empty_streams=os.environ["PROOF_MODE"] == "empty-stream")


@functools.wraps(original_step)
def step(*args, **kwargs):
    number = int(kwargs["iteration"]) + 1
    if number == 6:
        torch.distributed.barrier()
        torch.cuda.synchronize()
        torch.cuda.check_error(torch.cuda.cudart().cudaProfilerStart())
        trace_units.enabled = True
    result = original_step(*args, **kwargs)
    if number == 6:
        torch.cuda.synchronize()
        trace_units.enabled = False
        torch.cuda.check_error(torch.cuda.cudart().cudaProfilerStop())
        torch.distributed.barrier()
    steps.append({"step": number, "skipped": int(result[1]),
                  "loss": {k: v.item() for k, v in result[0].items()
                           if torch.is_tensor(v) and v.numel() == 1}})
    return result


training.train_step = step
try:
    runpy.run_path(str(source / "pretrain_gpt.py"), run_name="__main__")
finally:
    (output / f"ownership-rank-{rank}.json").write_text(json.dumps({
        "rank": rank, "steps": steps, "groups": trace_units.groups,
        "contexts": trace_units.contexts, "records": trace_units.records,
        "allocator_budget_bytes": budget, "serialization": serialization,
    }, indent=2) + "\n")
