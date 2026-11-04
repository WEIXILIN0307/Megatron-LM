"""Reproduce the Qwen TP2 dgrad failure with real NVFP4 TE Linear calls."""

import json
import os
from pathlib import Path

import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import NVFP4BlockScaling
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

assert os.getenv('NVTE_BACKWARD_OVERRIDE') is None
recipe = NVFP4BlockScaling()
assert not recipe.disable_rht and not recipe.disable_stochastic_rounding
results = []
for width in (4096, 4112, 4128):
    FP8GlobalStateManager.reset()
    layer = te.Linear(1024, width, bias=False, params_dtype=torch.bfloat16).cuda()
    inputs = torch.randn(128, 1024, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    try:
        with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
            loss = layer(inputs).float().square().mean()
        loss.backward()
        torch.cuda.synchronize()
        assert torch.isfinite(inputs.grad).all() and torch.isfinite(layer.weight.grad).all()
        result = {'width': width, 'outcome': 'passed'}
    except RuntimeError as error:
        assert width == 4112 and 'CUBLAS_STATUS_NOT_SUPPORTED' in str(error), str(error)
        result = {'width': width, 'outcome': 'failed', 'error': str(error)}
    results.append(result)
    print(json.dumps(result), flush=True)
    del layer, inputs, loss
Path(os.environ['NVFP4_SHAPE_OUTPUT']).write_text(json.dumps(results, indent=2) + '\n')
