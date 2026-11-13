"""Hash actual local tensors from the unchanged stock trainer without writing checkpoints.

This verification run is not a performance benchmark. Each digest reads every logical
element; 64 MiB chunks bound additional host memory. Matching digests are byte-level
evidence for the sampled states, not a claim about unobserved iterations or convergence.
"""

import functools
import hashlib
import inspect
import json
import os
import runpy
from pathlib import Path

import torch
from torch.distributed.tensor import DTensor

import megatron.training.training as training
from megatron.core.transformer.module import GraphableMegatronModule

output = Path(os.environ["FOLLOWUP_OUTPUT"])
source = Path(os.environ["FOLLOWUP_SOURCE"])
variant = os.environ["FOLLOWUP_VARIANT"]
rank = int(os.environ["RANK"])
original_step = training.train_step
signature = inspect.signature(original_step)
snapshots, steps = {}, []
replays = 0
original_replay = GraphableMegatronModule._te_cuda_graph_replay


def replay(self, *args, **kwargs):
    global replays
    replays += 1
    return original_replay(self, *args, **kwargs)


def digest(tensor):
    if tensor is None:
        return None
    if isinstance(tensor, DTensor):
        tensor = tensor.to_local()
    value = tensor.detach().reshape(-1)
    hasher = hashlib.sha256()
    width = max(1, 64 * 1024 * 1024 // value.element_size())
    for offset in range(0, value.numel(), width):
        chunk = value[offset:offset + width].contiguous().cpu().view(torch.uint8)
        hasher.update(memoryview(chunk.numpy()))
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
        "numel": tensor.numel(), "bytes": tensor.numel() * tensor.element_size(),
        "sha256": hasher.hexdigest()}


def parameters(models, gradients=False):
    return {f"chunk{index}/{name}": digest(param.grad if gradients else param)
        for index, model in enumerate(models) for name, param in model.named_parameters()}


def optimizer_snapshot(optimizer):
    leaves = getattr(optimizer, "chained_optimizers", None)
    if leaves is not None:
        return {str(index): optimizer_snapshot(leaf) for index, leaf in enumerate(leaves)}
    native = optimizer.optimizer
    result = {"parameters": {}, "state": {}, "groups": []}
    for group_index, group in enumerate(native.param_groups):
        result["groups"].append({key: str(value) for key, value in group.items() if key != "params"})
        for param_index, param in enumerate(group["params"]):
            key = f"group{group_index}/param{param_index}"
            result["parameters"][key] = digest(param)
            result["state"][key] = {name: digest(value) if torch.is_tensor(value) else value
                for name, value in native.state.get(param, {}).items()}
    return result


@functools.wraps(original_step)
def verified_step(*args, **kwargs):
    global replays
    bound = signature.bind_partial(*args, **kwargs).arguments
    step = int(bound["iteration"]) + 1
    models, optimizer = bound["model"], bound["optimizer"]
    if step in (4, 12):
        previous_observer = bound.get("tensor_metric_observer")

        def observe(**values):
            if previous_observer is not None:
                previous_observer(**values)
            snapshots[f"step{step}/gradients_before_optimizer"] = parameters(models, True)
            if os.environ.get("PROOF_GRADIENT_FOCUS") == "1" and rank == 1 and step == 12:
                target = "module.module.decoder.layers.13.mlp.experts.linear_fc2.weight3"
                parameter = dict(models[0].named_parameters())[target]
                gradient = parameter.grad
                if isinstance(gradient, DTensor):
                    gradient = gradient.to_local()
                torch.save(gradient.detach().cpu().clone(), output / "focused-gradient.pt")

        kwargs["tensor_metric_observer"] = observe
    replays = 0
    result = original_step(*args, **kwargs)
    torch.cuda.synchronize()
    assert int(result[1]) == 0
    if step in (4, 12):
        snapshots[f"step{step}/model_after_optimizer"] = parameters(models)
        snapshots[f"step{step}/optimizer_after_optimizer"] = optimizer_snapshot(optimizer)
        assert any(value is not None and value["numel"] for value in snapshots[f"step{step}/gradients_before_optimizer"].values())
    steps.append({"step": step, "forward_graph_replays": replays,
        "loss": {key: value.item() for key, value in result[0].items()
            if torch.is_tensor(value) and value.numel() == 1}})
    return result


training.train_step = verified_step
GraphableMegatronModule._te_cuda_graph_replay = replay
try:
    runpy.run_path(str(source / "pretrain_gpt.py"), run_name="__main__")
finally:
    (output / f"states-rank-{rank}.json").write_text(json.dumps({
        "rank": rank, "variant": variant, "steps": steps, "snapshots": snapshots,
        "method": "SHA256 of all logical local-shard bytes; no tensor sampling or checkpoint writes",
        "limits": "State verification only, no timing claim; snapshots at steps 4 and 12",
    }, indent=2) + "\n")
