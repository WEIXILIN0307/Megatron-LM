from pathlib import Path
import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

here = Path(__file__).resolve().parent
torch.manual_seed(2026)
tok = AutoTokenizer.from_pretrained(here / 'qwen3-teacher', local_files_only=True)
cfg = Qwen3Config(vocab_size=151680, hidden_size=128, intermediate_size=384,
                  num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                  head_dim=32, max_position_embeddings=256, tie_word_embeddings=True,
                  eos_token_id=tok.eos_token_id, bos_token_id=tok.bos_token_id,
                  pad_token_id=tok.pad_token_id)
model = Qwen3ForCausalLM(cfg).to(torch.bfloat16)
dest = here / 'tiny-qwen3'
model.save_pretrained(dest)
tok.save_pretrained(dest)
