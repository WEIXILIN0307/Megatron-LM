"""Retain existing train_step CPU return values; use existing training timers."""

import json
import os
import runpy
from pathlib import Path

if os.environ.get("DCP_MEASURE_CPU_BINDINGS"):
    bindings = json.loads(os.environ["DCP_MEASURE_CPU_BINDINGS"])
    os.sched_setaffinity(0, bindings[int(os.environ["LOCAL_RANK"])])
from megatron.training import training

rows = []
profile_rows = []
current_iteration = None
if os.environ.get("DCP_PROFILE_CPU"):
    import functools
    import gc
    import time

    from megatron.core.datasets import data_schedule
    from megatron.training.logging.packed_sequence_stats import DcpStatsAccumulator

    def profile(owner, name):
        function = getattr(owner, name)

        @functools.wraps(function)
        def measured(*args, **kwargs):
            start = time.perf_counter_ns()
            result = function(*args, **kwargs)
            profile_rows.append(
                dict(
                    function=name,
                    iteration=current_iteration,
                    microseconds=(time.perf_counter_ns() - start) / 1000,
                )
            )
            return result

        setattr(owner, name, measured)

    profile(data_schedule, "compute_schedule_stats")
    profile(training, "training_log")
    profile(DcpStatsAccumulator, "metrics")
    profile(DcpStatsAccumulator, "update")
    gc_start = {}

    def gc_callback(phase, info):
        generation = info["generation"]
        if phase == "start":
            gc_start[generation] = time.perf_counter_ns()
        elif generation in gc_start:
            profile_rows.append(
                dict(
                    function="gc_generation_" + str(generation),
                    iteration=current_iteration,
                    microseconds=(time.perf_counter_ns() - gc_start.pop(generation))
                    / 1000,
                )
            )

    gc.callbacks.append(gc_callback)

original = training.train_step


def step(*a, **kw):
    global current_iteration
    current_iteration = kw["iteration"] + 1
    result = original(*a, **kw)
    rows.append(
        dict(
            iteration=kw["iteration"] + 1,
            microbatches=result[12],
            tokens=result[9],
            squared_lengths=result[10],
            skipped=result[1],
        )
    )
    return result


training.train_step = step
try:
    runpy.run_path("pretrain_gpt.py", run_name="__main__")
finally:
    out = Path(os.environ["DCP_AUDIT_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    if os.environ.get("DCP_PROFILE_CPU"):
        gc.callbacks.remove(gc_callback)
        (out / f"cpu-profile-rank{os.environ['RANK']}.json").write_text(
            json.dumps(profile_rows, indent=2) + "\n"
        )
    (out / f"measure-rank{os.environ['RANK']}.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )
