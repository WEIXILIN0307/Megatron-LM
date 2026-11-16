"""One current-head EP2/expert-DP2 step, using the existing real-GPT MoE helpers."""

import json
import os
from dataclasses import replace
from pathlib import Path

import torch

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
from megatron.core import config as core_config

core_config.ENABLE_EXPERIMENTAL = True
from megatron.core.num_microbatches_calculator import init_num_microbatches_calculator
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.cuda_graphs import TECudaGraphHelper
from megatron.core.transformer.moe.moe_logging import destroy_moe_metrics_tracker
from tests.unit_tests.distributed.mfsdp_v2.test_mcore_partial_cuda_graph import (
    _build_moe_model, _moe_config, _train_moe_step,
)
from tests.unit_tests.test_utilities import Utils
import trace_units

Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=2)
os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
os.environ["NVTE_FUSED_ATTN"] = "0"
os.environ["NVTE_FLASH_ATTN"] = "0"
rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
assert world == 4
init_num_microbatches_calculator(rank=rank, global_batch_size=2 * world,
                               micro_batch_size=1, data_parallel_size=world)
groups = ProcessGroupCollection.use_mpu_process_groups()
config = replace(_moe_config(), hidden_size=512, kv_channels=128,
                 ffn_hidden_size=1024, moe_ffn_hidden_size=1024)
torch.manual_seed(771 + rank)
batches = [torch.randint(0, 128, (2, 1, 17), device="cuda") for _ in range(6)]
out = Path(os.environ["PROOF_OUTPUT"])


def run(graphs):
    model, optimizer = _build_moe_model(config, groups, False,
        ["attn", "mlp", "moe_router", "moe_preprocess"], "optim_grads_params", graphs)
    helper = TECudaGraphHelper([model], model.config, 16, 1, [optimizer], groups) if graphs else None
    losses, last = [], None
    try:
        for index, batch in enumerate(batches):
            if helper is not None and index == 3:
                helper.create_cudagraphs()
            profile = graphs and index == 4
            if profile:
                torch.distributed.barrier()
                torch.cuda.synchronize()
                torch.cuda.check_error(torch.cuda.cudart().cudaProfilerStart())
                trace_units.enabled = True
            last = _train_moe_step(model, optimizer, batch)
            if profile:
                torch.cuda.synchronize()
                trace_units.enabled = False
                torch.cuda.check_error(torch.cuda.cudart().cudaProfilerStop())
                torch.distributed.barrier()
            losses.append(last[0])
        return losses, last
    finally:
        if helper is not None and helper.graphs_created():
            helper.delete_cuda_graphs()
        destroy_moe_metrics_tracker()


try:
    actual, expected = run(True), run(False)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    result = {"rank": rank, "eager_graph_training_state_equal": True,
              "steps": 6, "profiled_step": 5,
              "model": "GPT MoE: 2 layers, hidden512, FFN1024, 4 experts/top2, seq16, GA2, BF16/ZeRO3",
              "limits": "Synthetic-token focused composition/concurrency probe; not a performance benchmark."}
    (out / f"correctness-rank-{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
finally:
    (out / f"ownership-rank-{rank}.json").write_text(json.dumps({
        "rank": rank, "groups": trace_units.groups, "contexts": trace_units.contexts,
        "records": trace_units.records, "capture_collectives": trace_units.capture_collectives,
    }, separators=(",", ":")) + "\n")
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
