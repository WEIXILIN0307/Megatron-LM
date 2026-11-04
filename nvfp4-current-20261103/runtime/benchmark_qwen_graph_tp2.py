"""Measure native BF16 training steps after real TE graph capture, in fresh processes."""

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import replace
from functools import partial
from pathlib import Path

import torch

from megatron.core import parallel_state
from megatron.core.dist_checkpointing.dict_utils import dict_list_map_outplace
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_layer_with_experimental_attention_variant_spec,
)
from megatron.core.models.hybrid.hybrid_block import HybridStack, HybridStackSubmodules
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.num_microbatches_calculator import (
    destroy_num_microbatches_calculator,
    init_num_microbatches_calculator,
)
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.pipeline_parallel.fine_grained_activation_offload import PipelineOffloadManager
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import (
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.cuda_graphs import TECudaGraphHelper
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.module import Float16Module
from megatron.core.utils import get_model_config, unwrap_model
from tests.unit_tests.ssm.test_gdn_offload_training import (
    _forward_step as _native_forward_step,
    _training_config,
)
from tools.ssm.qwen_gdn_offload_training import (
    _batches,
    _fingerprint,
    _forward_step,
    _provider,
    _tokenize,
)


def build_models(args, groups):
    graph = dict(
        cuda_graph_impl="transformer_engine",
        cuda_graph_modules=[],
        cuda_graph_warmup_steps=3,
        use_te_rng_tracker=True,
        delay_offload_until_cuda_graph=False,
        disable_parameter_transpose_cache=args.precision == "fp4" and not args.cached,
    )
    if args.model == "qwen":
        config = _provider(args)
        for name, value in graph.items():
            setattr(config, name, value)
        parents = config.provide_distributed_model(
            wrap_with_ddp=False, mixed_precision_wrapper=None, pg_collection=groups
        )
        raw = [parent.language_model for parent in parents]
    else:
        config = replace(
            _training_config(1, False, 1, 1, None, 1.0, False, variant="gdn2"),
            hidden_size=1024,
            ffn_hidden_size=3584,
            kv_channels=128,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_num_key_heads=8,
            linear_num_value_heads=16,
            attention_backend=AttnBackend.auto,
            min_offloaded_tensor_size=args.min_offloaded_tensor_size,
            fine_grained_activation_offloading=args.fraction is not None,
            offload_modules=["gdn_core_attn"] if args.fraction is not None else [],
            **graph,
        )
        specs = get_transformer_layer_with_experimental_attention_variant_spec(config)
        raw = [
            HybridModel(
                config=config,
                hybrid_stack_spec=ModuleSpec(
                    module=HybridStack,
                    submodules=HybridStackSubmodules(
                        gdn_layer=specs[0], gdn2_layer=specs[0], attention_layer=specs[3]
                    ),
                ),
                hybrid_layer_pattern="GGG*GGG*",
                vocab_size=4096,
                max_sequence_length=args.seq_length,
                pre_process=True,
                post_process=True,
                share_embeddings_and_output_weights=True,
                position_embedding_type="rope",
                rotary_percent=0.25,
                rotary_base=10_000_000,
                pg_collection=groups,
            ).cuda()
        ]
    models = [
        DistributedDataParallel(
            config,
            DistributedDataParallelConfig(grad_reduce_in_fp32=True),
            Float16Module(config, model),
            pg_collection=groups,
        )
        for model in raw
    ]
    for model in models:
        model.broadcast_params()
    optimizer = get_megatron_optimizer(
        OptimizerConfig(bf16=True, lr=1e-4, clip_grad=1.0),
        models,
        use_gloo_process_groups=False,
        pg_collection=groups,
    )
    return models, optimizer


def run(args):
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group("nccl")
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=2)
    initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
    init_num_microbatches_calculator(
        rank=0,
        global_batch_size=args.microbatches,
        micro_batch_size=1,
        data_parallel_size=1,
        decrease_batch_size_if_needed=False,
    )
    groups = ProcessGroupCollection.use_mpu_process_groups()
    torch.manual_seed(31)
    model_parallel_cuda_manual_seed(31)
    off_interface.reset_instance()
    helper = None
    try:
        token_ids = (
            _tokenize(args)
            if args.model == "qwen"
            else torch.randint(0, 4096, (65537,), generator=torch.Generator().manual_seed(19))
        )
        stream = torch.cuda.current_stream()
        init_stream = torch.cuda.Stream()
        init_stream.wait_stream(stream)
        with torch.cuda.stream(init_stream):
            models, optimizer = build_models(args, groups)
        stream.wait_stream(init_stream)
        torch.cuda.empty_cache()
        config = get_model_config(models[0])
        manager = PipelineOffloadManager.get_instance()
        pool_checks = []

        def finalize(*positional, **kwargs):
            finalize_model_grads(*positional, **kwargs)
            status = manager.cpu_tensor_pool.get_pool_status()
            assert status["global_stats"]["current_in_use"] == 0
            pool_checks.append(status["global_stats"])

        config.grad_scale_func = optimizer.scale_loss
        config.finalize_model_grads_func = partial(finalize, pg_collection=groups)
        schedule = get_forward_backward_func(pp_size=1, vp_size=None)
        p2p = P2PCommunicator(groups.pp, config)
        records = []
        captures = []
        for step in range(3 + args.warmup + args.iterations):
            if step == 3 and not args.no_graph:
                torch.cuda.synchronize()
                begin = time.perf_counter()
                helper = TECudaGraphHelper(
                    models, config, args.seq_length, 1, optimizers=[optimizer], pg_collection=groups
                )
                helper.create_cudagraphs()
                torch.cuda.synchronize()
                assert helper.graphs_created()
                captures = [len(layer.cuda_graphs) for layer in unwrap_model(models[0]).decoder.layers]
                assert all(count == helper.num_microbatches for count in captures), captures
                print(json.dumps({"capture_seconds": time.perf_counter() - begin, "graphs": captures}), flush=True)
            batches = _batches(token_ids, args, step, groups)
            input_hash = hashlib.sha256(b"".join(batch["tokens"].cpu().numpy().tobytes() for batch in batches)).hexdigest()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            begin = time.perf_counter()
            optimizer.zero_grad()
            for model in models:
                model.zero_grad_buffer()
            losses = schedule(
                forward_step_func=_forward_step if args.model == "qwen" else _native_forward_step,
                data_iterator=[iter(batches)],
                model=models,
                num_microbatches=args.microbatches,
                seq_length=args.seq_length,
                micro_batch_size=1,
                forward_only=False,
                p2p_communicator=p2p,
                pg_collection=groups,
            )
            gradients = (
                {name: _fingerprint(p.main_grad) for name, p in models[0].named_parameters()}
                if args.check_state else None
            )
            success, grad_norm, _ = optimizer.step()
            torch.cuda.synchronize()
            seconds = time.perf_counter() - begin
            record = dict(
                step=step,
                replay=step >= 3 and not args.no_graph,
                measured=not args.check_state and step >= 3 + args.warmup,
                seconds=seconds,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                allocated_bytes=torch.cuda.memory_allocated(),
                reserved_bytes=torch.cuda.memory_reserved(),
                input_sha256=input_hash,
                loss=[float(item["loss"]) for item in losses],
                grad_norm=float(grad_norm),
                pool=pool_checks[-1],
                selected_bytes=manager.offload_summary_total_bytes if args.fraction is not None else 0,
            )
            assert success and math.isfinite(grad_norm) and grad_norm > 0
            assert all(math.isfinite(loss) for loss in record["loss"])
            if args.fraction is not None and step >= 3:
                assert record["selected_bytes"] > 0
            if args.check_state:
                record.update(
                    gradients=gradients,
                    weights={name: _fingerprint(p) for name, p in models[0].named_parameters()},
                    optimizer=dict_list_map_outplace(_fingerprint, optimizer.state_dict()),
                )
            records.append(record)
            print(json.dumps({k: v for k, v in record.items() if k not in ("gradients", "weights", "optimizer")}), flush=True)
        result = dict(
            arguments=vars(args),
            torch=torch.__version__, cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(), graph_counts=captures,
            helper_microbatches=helper.num_microbatches if helper is not None else None,
            parameters=sum(p.numel() for model in models for p in model.parameters()),
            config=repr(config), dataset_tokens=len(token_ids), records=records,
            pool_checks=len(pool_checks),
        )
        args.output.with_suffix(f".rank{torch.distributed.get_rank()}.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    finally:
        if helper is not None and helper.graphs_created():
            helper.delete_cuda_graphs()
        torch.cuda.synchronize()
        off_interface.reset_instance()
        destroy_num_microbatches_calculator()
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    from megatron.core.transformer import ModuleSpec

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["qwen", "gdn2"], required=True)
    parser.add_argument("--precision", choices=["bf16", "fp4"], default="bf16")
    parser.add_argument("--threshold", type=int, default=1024 * 1024)
    parser.add_argument("--cached", action="store_true")
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--check-state", action="store_true")
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    options = parser.parse_args()
    options.fraction = 1.0 if options.offload else None
    options.tp, options.sp, options.cp, options.pp, options.vp = 2, True, 1, 1, None
    options.microbatches = 4
    options.fused_pre_gdr, options.recompute_norm = False, False
    options.min_offloaded_tensor_size = options.threshold
    options.max_documents = 128
    os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
    run(options)
