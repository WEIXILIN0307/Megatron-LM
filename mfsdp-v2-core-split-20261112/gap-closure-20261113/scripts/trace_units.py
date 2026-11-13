"""Add trace-only ownership labels around existing operations; do not add CUDA work."""

import functools
import itertools
import threading
from contextlib import contextmanager

import torch
from transformer_engine.pytorch import graph as te_graph

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.context import FsdpContext
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.parameter_group import FsdpParameterGroup
from megatron.core.tensor_parallel.mappings import _AllToAll
from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
from megatron.core.transformer.moe import token_dispatcher

enabled = False
records = []
groups = {}
contexts = {}
capture_collectives = []
generations = {}
sequence = itertools.count(1)
local = threading.local()


def phase_name(phase):
    return phase.name.lower() if phase is not None else "both"


def group_id(group):
    key = str(id(group))
    if key not in groups:
        owner = group._owning_module()
        context = owner.context
        context_key = str(id(context))
        if context_key not in contexts:
            contexts[context_key] = {
                "order": [{"unit": event.module.name or "<root>", "phase": phase_name(event.phase)}
                          for event in context.execution_order],
                "streams": {name: int(getattr(context, name).cuda_stream) for name in (
                    "allgather_stream", "inner_reduction_stream", "outer_reduction_stream")},
            }
        groups[key] = {
            "unit": owner.name or "<root>",
            "context": context_key,
            "group_index": next(i for i, g in enumerate(owner.parameter_groups) if g is group),
            "mesh_ranks": group.mesh.mesh.tolist(),
            "mesh_dim_names": group.mesh.mesh_dim_names,
            "placements": [str(p) for p in group.model_weight.placements],
            "parameter_names": [name for p in group.fsdp_parameters for name in p.fqns],
        }
    return key


@contextmanager
def marked(kind, **fields):
    record = {"kind": kind, "id": next(sequence), **fields}
    label = "proof|" + kind + "|" + "|".join(f"{k}={v}" for k, v in record.items() if k != "kind")
    with torch.cuda.nvtx.range(label):
        yield record
    records.append(record)


original_unshard = FsdpParameterGroup.unshard_parameters
original_wait = FsdpParameterGroup.wait_for_unshard
original_gather = FsdpParameterGroup._allgather_model_weight
original_reduce = FsdpParameterGroup.reduce_inner_gradients
original_call = FsdpModule._call_cuda_graph
original_replay = torch.cuda.CUDAGraph.replay
original_pack_reduce = FsdpModule._reduce_inner_gradients
original_finish = FsdpContext.finish_grad_sync


@functools.wraps(original_unshard)
def unshard(self, phase=None):
    if not enabled:
        return original_unshard(self, phase)
    gid = group_id(self)
    before = {int(event.cuda_event) for event in self._unshard_events.values()}
    with marked("unshard", group=gid, unit=groups[gid]["unit"], phase=phase_name(phase)) as row:
        context = self._owning_module().context
        cursor = context.execution_order.peek()
        row["cursor"] = ({"position": context.execution_order.position,
                          "unit": cursor.module.name or "<root>", "phase": phase_name(cursor.phase)}
                         if cursor is not None else None)
        row["stream_native"] = int(torch.cuda.current_stream().cuda_stream)
        previous = getattr(local, "unshard", None)
        local.unshard = row["id"]
        try:
            result = original_unshard(self, phase)
        finally:
            local.unshard = previous
        handles = {int(event.cuda_event) for event in self._unshard_events.values()}
        for handle in handles - before:
            generations[handle] = row["id"]
        row["readiness"] = [{"phase": phase_name(p), "handle": int(event.cuda_event),
                             "producer": generations.get(int(event.cuda_event))}
                            for p, event in self._unshard_events.items()]
    return result


@functools.wraps(original_wait)
def wait(self, phase):
    if not enabled:
        return original_wait(self, phase)
    gid = group_id(self)
    requested = list(self._unshard_events) if phase is None else [phase]
    handles = list(dict.fromkeys(int(self._unshard_events[p].cuda_event) for p in requested))
    with marked("wait", group=gid, unit=groups[gid]["unit"], phase=phase_name(phase)) as row:
        row["stream_native"] = int(torch.cuda.current_stream().cuda_stream)
        row["readiness"] = [{"handle": handle, "producer": generations.get(handle)} for handle in handles]
        return original_wait(self, phase)


@functools.wraps(original_gather)
def gather(self, *args, **kwargs):
    if te_graph.is_graph_capturing():
        capture_collectives.append({"kind": "FSDP_parameter_AG",
                                    "cuda_capturing": torch.cuda.is_current_stream_capturing(),
                                    "unit": self._owning_module().name,
                                    "ranks": self.mesh.mesh.tolist()})
    if not enabled:
        return original_gather(self, *args, **kwargs)
    gid = group_id(self)
    with marked("AG", group=gid, unit=groups[gid]["unit"], producer=getattr(local, "unshard", None)):
        return original_gather(self, *args, **kwargs)


@functools.wraps(original_reduce)
def reduce(self, *args, **kwargs):
    if not enabled:
        return original_reduce(self, *args, **kwargs)
    gid = group_id(self)
    with marked("RS", group=gid, unit=groups[gid]["unit"]):
        return original_reduce(self, *args, **kwargs)


@functools.wraps(original_call)
def graph_call(self, *args, **kwargs):
    if te_graph.is_graph_capturing():
        previous = getattr(local, "capture_unit", None)
        local.capture_unit = self.name
        try:
            return original_call(self, *args, **kwargs)
        finally:
            local.capture_unit = previous
    if not enabled:
        return original_call(self, *args, **kwargs)
    with marked("graph_call", unit=self.name):
        return original_call(self, *args, **kwargs)


@functools.wraps(original_replay)
def replay(self):
    if not enabled:
        return original_replay(self)
    with marked("replay", stream_native=int(torch.cuda.current_stream().cuda_stream)):
        return original_replay(self)


@functools.wraps(original_pack_reduce)
def pack_reduce(self, group):
    if not enabled:
        return original_pack_reduce(self, group)
    gid = group_id(group)
    with marked("pack_reduce", group=gid, unit=groups[gid]["unit"]):
        return original_pack_reduce(self, group)


@functools.wraps(original_finish)
def finish(self):
    if not enabled:
        return original_finish(self)
    with marked("finish_grad_sync", stream_native=int(torch.cuda.current_stream().cuda_stream)):
        return original_finish(self)


def wrap_dispatcher(name, stage):
    original = getattr(MoEAlltoAllTokenDispatcher, name)

    @functools.wraps(original)
    def call(self, *args, **kwargs):
        if not enabled:
            return original(self, *args, **kwargs)
        previous = getattr(local, "stage", None)
        local.stage = stage
        try:
            return original(self, *args, **kwargs)
        finally:
            local.stage = previous

    setattr(MoEAlltoAllTokenDispatcher, name, call)


original_a2a_forward = _AllToAll.forward
original_a2a_backward = _AllToAll.backward
original_routing_gather = token_dispatcher.gather_from_sequence_parallel_region


@functools.wraps(original_routing_gather)
def routing_gather(input_, *args, **kwargs):
    if te_graph.is_graph_capturing():
        group = kwargs.get("group")
        assert group is not None
        capture_collectives.append({
            "kind": "token_dispatcher_gather", "unit": getattr(local, "capture_unit", None),
            "cuda_capturing": torch.cuda.is_current_stream_capturing(),
            "shape": list(input_.shape), "dtype": str(input_.dtype),
            "ranks": torch.distributed.get_process_group_ranks(group),
        })
    return original_routing_gather(input_, *args, **kwargs)


def a2a_forward(ctx, group, *args, **kwargs):
    if not enabled:
        return original_a2a_forward(ctx, group, *args, **kwargs)
    stage = getattr(local, "stage", None) or "unspecified"
    ctx._proof_stage = stage
    with marked("A2A", stage=stage) as row:
        row["ranks"] = torch.distributed.get_process_group_ranks(group)
        return original_a2a_forward(ctx, group, *args, **kwargs)


def a2a_backward(ctx, *args, **kwargs):
    if not enabled:
        return original_a2a_backward(ctx, *args, **kwargs)
    previous = getattr(local, "stage", None)
    local.stage = getattr(ctx, "_proof_stage", "unspecified").replace("_forward", "_backward")
    try:
        return original_a2a_backward(ctx, *args, **kwargs)
    finally:
        local.stage = previous


FsdpParameterGroup.unshard_parameters = unshard
FsdpParameterGroup.wait_for_unshard = wait
FsdpParameterGroup._allgather_model_weight = gather
FsdpParameterGroup.reduce_inner_gradients = reduce
FsdpModule._call_cuda_graph = graph_call
torch.cuda.CUDAGraph.replay = replay
FsdpModule._reduce_inner_gradients = pack_reduce
FsdpContext.finish_grad_sync = finish
wrap_dispatcher("token_dispatch", "dispatch_forward")
wrap_dispatcher("token_combine", "combine_forward")
_AllToAll.forward = staticmethod(a2a_forward)
_AllToAll.backward = staticmethod(a2a_backward)
token_dispatcher.gather_from_sequence_parallel_region = routing_gather
