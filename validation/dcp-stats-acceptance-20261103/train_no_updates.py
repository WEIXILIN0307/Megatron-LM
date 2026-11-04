"""Acceptance-only check of the upstream loop-without-updates path."""

import hashlib
import json
import os
import runpy
from pathlib import Path

import megatron.training as entry
import torch
from megatron.core.datasets import data_schedule
from megatron.training import training
from megatron.training.logging.packed_sequence_stats import DcpStatsAccumulator

out = Path(os.environ["DCP_AUDIT_DIR"])
out.mkdir(parents=True, exist_ok=True)
record = {"data_hook_iterations": [], "empty_submissions": 0, "empty_log_windows": 0}


def forbidden(*args, **kwargs):
    raise AssertionError(
        "No-update loop must not execute train_step or compute schedule statistics"
    )


training.train_step = forbidden
data_schedule.compute_schedule_stats = forbidden
original_update = DcpStatsAccumulator.update
original_metrics = DcpStatsAccumulator.metrics


def update(self, stats, skipped=False):
    assert stats == {}, stats
    record["empty_submissions"] += 1
    return original_update(self, stats, skipped)


def metrics(self, reset=True):
    result = original_metrics(self, reset)
    assert result == {}, result
    record["empty_log_windows"] += 1
    return result


DcpStatsAccumulator.update = update
DcpStatsAccumulator.metrics = metrics
original_train = training.train


def digest(model):
    h = hashlib.sha256()
    for chunk in model:
        for name, param in chunk.named_parameters():
            h.update(name.encode())
            h.update(
                param.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
            )
    return h.hexdigest()


def train(*args, **kwargs):
    assert kwargs["run_train_loop_without_updates"] is True
    assert args[2] is None, "No optimizer should be constructed"
    before = digest(args[1])
    hook = kwargs["next_train_iterator"]

    def next_iterator(iteration, iterator):
        record["data_hook_iterations"].append(iteration)
        return hook(iteration, iterator)

    kwargs["next_train_iterator"] = next_iterator
    result = original_train(*args, **kwargs)
    after = digest(args[1])
    assert before == after
    assert record["data_hook_iterations"] == [0, 1, 2]
    assert record["empty_submissions"] == record["empty_log_windows"] == 3
    record.update(parameter_hash=after, parameters_unchanged=True, iterations=result[0])
    (out / f"rank{os.environ['RANK']}.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    return result


training.train = train
original_pretrain = entry.pretrain


def pretrain(*args, **kwargs):
    kwargs["run_train_loop_without_updates"] = True
    return original_pretrain(*args, **kwargs)


entry.pretrain = pretrain
runpy.run_path("pretrain_gpt.py", run_name="__main__")
