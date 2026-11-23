"""Build an offline draft with the tiny vLLM teacher's actual embedding and norm."""
import sys
from copy import deepcopy
from pathlib import Path

import modelopt.torch.speculative as mtsp
import torch
from transformers import Qwen3ForCausalLM

from megatron.post_training.arguments import add_modelopt_args
from megatron.post_training.model_config import build_inference_models, modelopt_inference_config_from_args
from megatron.training.arguments import parse_and_validate_args
from megatron.training.checkpointing import save_checkpoint
from megatron.training.global_vars import set_run_config
from megatron.training.initialize import initialize_megatron

args = parse_and_validate_args(extra_args_provider=add_modelopt_args)
cfg = modelopt_inference_config_from_args(args)
set_run_config(cfg)
initialize_megatron(training=False)
[model] = build_inference_models(cfg, mixed_precision=False)
teacher = Qwen3ForCausalLM.from_pretrained(args.tokenizer_model, local_files_only=True)
with torch.no_grad():
    model.embedding.word_embeddings.weight.copy_(teacher.model.embed_tokens.weight)
    model.decoder.final_layernorm.weight.copy_(teacher.model.norm.weight)
    if model.output_layer.weight is not None:
        model.output_layer.weight.copy_(teacher.model.embed_tokens.weight)
draft = deepcopy(mtsp.config.EAGLE3_DEFAULT_CFG)
draft["config"]["eagle_offline"] = True
draft["config"]["eagle_architecture_config"].update(
    intermediate_size=384, num_attention_heads=4, num_key_value_heads=2,
    eagle_aux_hidden_state_layer_ids=[0, 1, 2],
)
model = mtsp.convert(model, draft).cuda().bfloat16()
assert model.eagle_offline
save_checkpoint(1, [model], None, None, 0, release=True)
torch.distributed.destroy_process_group()
