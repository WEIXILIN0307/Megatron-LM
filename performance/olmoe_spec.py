"""Validation-only OLMoE-derived 16-expert proxy; not the full public model.

Public source architecture: allenai/OLMoE-1B-7B-0924 at
6d84c48581ece794365f2b8e9cfb043c68ade9c5. No pretrained weights are loaded. The benchmark reduces experts from 64 to 16; it is a proxy, not the complete published model.
The only adapter restores OLMoE's full-width Q/K RMSNorm at MCore's
post-head-reshape normalization boundary. All computation uses existing TENorm.
"""

from copy import deepcopy

import torch

from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec


class FullWidthQKNorm(torch.nn.Module):
    def __init__(self, hidden_size, config, eps):
        super().__init__()
        assert config.tensor_model_parallel_size == 1
        assert config.num_attention_heads == config.num_query_groups == 16
        assert hidden_size * config.num_attention_heads == config.hidden_size == 2048
        self.norm = TENorm(config, config.hidden_size, eps)

    @property
    def weight(self):
        return self.norm.weight

    def forward(self, value):
        return self.norm(value.flatten(-2)).reshape_as(value)


def olmoe_stack_spec(config):
    assert config.num_layers == 32
    assert config.num_moe_experts == 16
    assert config.moe_router_topk == 8
    assert config.moe_ffn_hidden_size == 1024
    assert config.layernorm_epsilon == 1e-5
    assert config.qk_layernorm and config.moe_router_pre_softmax
    assert config.normalization == "RMSNorm"
    spec = deepcopy(hybrid_stack_spec)
    attention = spec.submodules.attention_layer.submodules.self_attention.submodules
    attention.q_layernorm = FullWidthQKNorm
    attention.k_layernorm = FullWidthQKNorm
    return spec
