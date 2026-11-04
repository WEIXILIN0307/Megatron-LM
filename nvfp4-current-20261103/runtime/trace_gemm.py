"""Log a failing TE GEMM's operands, then propagate the original exception."""

import importlib
import json
import runpy
import sys

import torch  # Load the wheel's CUDA/NCCL dependencies before Transformer Engine.

module = importlib.import_module('transformer_engine.pytorch.module.layernorm_linear')
original = module.general_gemm


def traced_gemm(*args, **kwargs):
    try:
        return original(*args, **kwargs)
    except RuntimeError:
        print('FAILED_GEMM', json.dumps({
            'operands': [{'type': type(t).__name__, 'shape': list(t.shape)} for t in args[:2]],
            'layout': kwargs.get('layout'), 'grad': kwargs.get('grad'),
            'out_dtype': str(kwargs.get('out_dtype')),
            'out_is_none': kwargs.get('out') is None,
        }), flush=True)
        raise


module.general_gemm = traced_gemm
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
