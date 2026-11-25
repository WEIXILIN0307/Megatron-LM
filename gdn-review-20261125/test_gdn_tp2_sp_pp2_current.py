"""Exercise current GDN native tests with TP2 + SP + PP2 on four ranks."""

import pytest
import torch

from tests.unit_tests.ssm.test_gdn_offload_cuda_graphs import (
    test_gdn_offload_with_te_graphs as run_graph,
)
from tests.unit_tests.ssm.test_gdn_offload_training import test_gdn_offload_training as run_eager
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize(
    ("mode", "variant"),
    [("eager", "gdn"), ("eager", "gdn2"), ("graph", "gdn"), ("graph", "gdn2")],
    ids=["eager-gdn", "eager-gdn2", "graph-gdn-attention", "graph-gdn2-layer"],
)
def test_tp2_sp_pp2(mode, variant, monkeypatch):
    assert Utils.world_size == 4
    torch.cuda.reset_peak_memory_stats()
    try:
        if mode == "eager":
            run_eager(
                tp=2,
                sp=True,
                cp=1,
                pp=2,
                vp=None,
                fraction=1.0,
                recompute_norm=False,
                fused_pre_gdr=False,
                variant=variant,
                native_hybrid=False,
                monkeypatch=monkeypatch,
            )
        else:
            run_graph(
                variant=variant,
                fused=False,
                recompute_norm=False,
                whole_layer=variant == "gdn2",
                delay=variant == "gdn2",
                fraction=1.0,
                tp=2,
                pp=2,
                mixed=False,
                monkeypatch=monkeypatch,
            )
    finally:
        mib = 1024**2
        print(
            f"GPU_MEMORY mode={mode} variant={variant} "
            f"peak_allocated_mib={torch.cuda.max_memory_allocated()/mib:.1f} "
            f"peak_reserved_mib={torch.cuda.max_memory_reserved()/mib:.1f}",
            flush=True,
        )
