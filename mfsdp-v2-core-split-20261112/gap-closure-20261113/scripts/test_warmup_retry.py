"""Dependency qualification: real MoE capture fails inside TE warmup, then retries."""

from dataclasses import replace

import pytest
import torch
from transformer_engine.pytorch import graph as te_graph

from megatron.core.transformer import cuda_graphs
from megatron.core.transformer.cuda_graphs import TECudaGraphHelper
from tests.unit_tests.distributed.mfsdp_v2.test_mcore_partial_cuda_graph import (
    _build_moe_model,
    _moe_config,
    _rng_contents,
    _train_moe_step,
    process_groups,
)


@pytest.fixture(scope="session", autouse=True)
def ensure_test_data():
    """This focused model probe does not consume the suite's external test assets."""


@pytest.mark.parametrize("hybrid", [False, True], ids=["gpt", "hybrid"])
def test_moe_te_warmup_failure_retry(process_groups, hybrid, monkeypatch):
    config = replace(_moe_config(hybrid=hybrid), attention_dropout=0.1, hidden_dropout=0.1)
    scopes = ["attn", "mlp", "moe_router", "moe_preprocess"]
    torch.manual_seed(92)
    batches = [torch.randint(0, 128, (2, 1, 17), device="cuda") for _ in range(4)]

    def run(graphs):
        model, optimizer = _build_moe_model(config, process_groups, hybrid, scopes,
                                           "optim_grads_params", graphs)
        helper = (TECudaGraphHelper([model], model.config, 16, 1, [optimizer], process_groups)
                  if graphs else None)
        try:
            for batch in batches[:3]:
                _train_moe_step(model, optimizer, batch)
            if helper is not None:
                before_rng = _rng_contents()
                layers = helper.flattened_callables
                original_calls = {type(layer): type(layer).__call__ for layer in layers}
                names = ("_forward_pre_hooks", "_forward_hooks", "_backward_pre_hooks", "_backward_hooks")
                hooks = [(module, {name: dict(getattr(module, name)) for name in names})
                         for module in model.modules()]
                attention = next(module for layer in layers for module in layer.modules()
                                 if hasattr(module, "linear_qkv"))
                target = attention.linear_qkv
                original_forward = target.forward
                failures = []

                def fail_after_qkv(*args, **kwargs):
                    result = original_forward(*args, **kwargs)
                    assert te_graph.is_graph_capturing()
                    torch.rand(1, device="cuda")
                    failures.append(True)
                    raise RuntimeError("controlled TE warmup failure")

                with monkeypatch.context() as patch:
                    patch.setattr(target, "forward", fail_after_qkv)
                    with pytest.raises(RuntimeError, match="controlled TE warmup failure"):
                        helper.create_cudagraphs()
                assert len(failures) == 1
                assert not cuda_graphs.is_graph_capturing()
                assert not te_graph.is_graph_capturing()
                assert not helper.graphs_created()
                torch.testing.assert_close(_rng_contents(), before_rng, rtol=0, atol=0)
                for module_type, original_call in original_calls.items():
                    assert module_type.__call__ is original_call
                for module, original_hooks in hooks:
                    for name in names:
                        assert getattr(module, name) == original_hooks[name], (module, name)
                for layer in layers:
                    assert not layer._fsdp_cuda_graph_capturing
                    assert not layer.cuda_graphs
                    assert not any(g._retain_unsharded_storage for g in layer.parameter_groups)
                helper.create_cudagraphs()
            return _train_moe_step(model, optimizer, batches[-1])
        finally:
            if helper is not None and helper.graphs_created():
                helper.delete_cuda_graphs()
            cuda_graphs._set_capture_end()

    torch.testing.assert_close(run(True), run(False), rtol=0, atol=0)
