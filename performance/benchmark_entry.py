"""External measurement wrapper; production runtime and numerical kernels are unchanged."""
import functools
import json
import os
import runpy
import sys
import time
from pathlib import Path

import torch
import megatron.training.training as training
from megatron.core.transformer.cuda_graphs import TECudaGraphHelper

root = Path(os.environ['MFSDP_BENCH_OUTPUT'])
rank = int(os.environ['RANK'])
rows = []
captures = []
original_step = training.train_step
original_capture = TECudaGraphHelper.create_cudagraphs

@functools.wraps(original_step)
def measured_step(*args, **kwargs):
    step = int(kwargs['iteration']) + 1
    torch.cuda.synchronize()
    if step == 16:
        torch.cuda.reset_peak_memory_stats()
    begin = time.perf_counter()
    result = original_step(*args, **kwargs)
    torch.cuda.synchronize()
    duration = time.perf_counter() - begin
    loss = {}
    for key, value in result[0].items():
        if torch.is_tensor(value) and value.numel() == 1:
            loss[key] = value.item()
    rows.append({'step': step, 'seconds': duration, 'start_counter': begin, 'end_counter': begin + duration, 'loss': loss,
                 'skipped': int(result[1]),
                 'allocated_bytes': torch.cuda.memory_allocated(),
                 'reserved_bytes': torch.cuda.memory_reserved(),
                 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                 'peak_reserved_bytes': torch.cuda.max_memory_reserved()})
    return result

@functools.wraps(original_capture)
def measured_capture(self, *args, **kwargs):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = original_capture(self, *args, **kwargs)
    torch.cuda.synchronize()
    captures.append({'seconds': time.perf_counter() - start,
                     'allocated_bytes': torch.cuda.memory_allocated(),
                     'reserved_bytes': torch.cuda.memory_reserved()})
    return result

training.train_step = measured_step
TECudaGraphHelper.create_cudagraphs = measured_capture
try:
    runpy.run_path(os.environ['MFSDP_BENCH_SOURCE'] + '/pretrain_hybrid.py', run_name='__main__')
finally:
    (root / f'measurements-rank-{rank}.json').write_text(json.dumps({
        'rank': rank, 'steps': rows, 'capture': captures,
        'method': 'perf_counter around stock train_step with CUDA synchronization at boundaries; no per-step distributed barrier; max rank duration computed offline; steady steps 16-60',
    }, indent=2) + '\n')
