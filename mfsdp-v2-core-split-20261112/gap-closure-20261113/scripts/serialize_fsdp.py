"""Measurement control: serialize only CUDA Graph compute against FSDP streams."""

import functools
import time

import torch
from transformer_engine.pytorch import graph as te_graph

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule


def install(report, empty_streams=False):
    original_call = FsdpModule._call_cuda_graph
    original_replay = torch.cuda.CUDAGraph.replay
    contexts = {}
    dummy_streams = None
    report["serialization_policy"] = ("graph-boundaries-empty-stream-control"
                                      if empty_streams else "graph-boundaries-v2")
    report["serialized_graph_replays"] = 0
    report["barrier_cpu_seconds"] = 0.0

    @functools.wraps(original_call)
    def call(self, *args, **kwargs):
        contexts[id(self.context)] = self.context
        return original_call(self, *args, **kwargs)

    @functools.wraps(original_replay)
    def replay(self):
        nonlocal dummy_streams
        if not contexts or te_graph.is_graph_capturing():
            return original_replay(self)
        start = time.perf_counter()
        compute = torch.cuda.current_stream()
        streams = dict.fromkeys(stream for context in contexts.values() for stream in (
            context.allgather_stream, context.inner_reduction_stream, context.outer_reduction_stream))
        if empty_streams:
            if dummy_streams is None:
                dummy_streams = tuple(torch.cuda.Stream() for _ in streams)
            assert len(dummy_streams) == len(streams)
            streams = dummy_streams
        report["control_stream_count"] = len(streams)
        for stream in streams:
            compute.wait_stream(stream)
        before_replay = time.perf_counter()
        result = original_replay(self)
        after_replay = time.perf_counter()
        for stream in streams:
            stream.wait_stream(compute)
        report["serialized_graph_replays"] += 1
        if report.get("measure_active", True):
            report["barrier_cpu_seconds"] += time.perf_counter() - start - (after_replay - before_replay)
        return result

    FsdpModule._call_cuda_graph = call
    torch.cuda.CUDAGraph.replay = replay
