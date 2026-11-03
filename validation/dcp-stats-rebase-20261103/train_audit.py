"""Acceptance-only audit of real pretrain_gpt training; no schedule changes."""

import hashlib
import json
import os
import runpy
from pathlib import Path

import torch

from megatron.core.datasets import data_schedule
from megatron.core.datasets.data_schedule_utils import compute_schedule_stats
from megatron.training import training

out = Path(os.environ["DCP_AUDIT_DIR"])
out.mkdir(parents=True, exist_ok=True)
rank = int(os.environ["RANK"])
phase = "initialization"


def write(record):
    with (out / f"rank{rank}.jsonl").open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


write(
    dict(
        kind="source",
        hashes={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path("megatron/core/datasets/data_schedule.py"),
                Path("megatron/core/datasets/data_schedule_utils.py"),
                Path("megatron/training/training.py"),
                Path("megatron/training/logging/packed_sequence_stats.py"),
                Path("megatron/training/config/training_config.py"),
            ]
        },
    )
)
original = data_schedule.DefaultDynamicCPScheduler.get_groups_and_subsamples


def groups(self, lengths):
    result = original(self, lengths)
    unaligned_scheduler = data_schedule.DefaultDynamicCPScheduler(
        self.max_seqlen_per_dp_cp_rank,
        self.cp_size,
        self.dp_size,
        None,
        min_cp_size=self.min_cp_size,
        cp_group_sizes=self.cp_group_sizes,
        pad_sequences=self.pad_sequences,
        max_num_seqs=self.max_num_seqs,
    )
    unaligned = original(unaligned_scheduler, lengths)
    write(
        dict(
            kind="schedule",
            phase=phase,
            schedule=result,
            unaligned_microbatches=len(unaligned),
            lengths=lengths,
            stats=compute_schedule_stats(result, lengths),
        )
    )
    return result


data_schedule.DefaultDynamicCPScheduler.get_groups_and_subsamples = groups


def digest(model, attr=None):
    h = hashlib.sha256()
    for chunk in model:
        for name, parameter in chunk.named_parameters():
            tensor = getattr(parameter, attr, None) if attr else parameter
            if tensor is not None:
                h.update(name.encode())
                h.update(
                    tensor.detach()
                    .contiguous()
                    .view(torch.uint8)
                    .cpu()
                    .numpy()
                    .tobytes()
                )
    return h.hexdigest()


step_original = training.train_step


def step(*args, **kwargs):
    global phase
    phase = "train"
    result = step_original(*args, **kwargs)
    phase = "outside_train"
    model = args[2]
    write(
        dict(
            kind="step",
            iteration=kwargs["iteration"],
            losses={k: v.item() for k, v in result[0].items()},
            skipped=result[1],
            microbatches=result[12],
            params=digest(model),
            grads=digest(model, "main_grad"),
            stats=kwargs.get("schedule_stats"),
        )
    )
    return result


training.train_step = step
evaluate_original = training.evaluate


def evaluate(*a, **kw):
    global phase
    phase = "eval"
    result = evaluate_original(*a, **kw)
    phase = "outside_train"
    return result


training.evaluate = evaluate
runpy.run_path("pretrain_gpt.py", run_name="__main__")
