"""Run the existing full-size OLMoE recipe with ordinary configurable dropout."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("name")
parser.add_argument("--graphs", action="store_true")
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--exit-step", type=int)
parser.add_argument("--load")
parser.add_argument("--ckpt-step", type=int)
parser.add_argument("--save-interval", type=int, default=250)
parser.add_argument("--eval-interval", type=int, default=250)
parser.add_argument("--eval-iters", type=int, default=10)
parser.add_argument("--sequence-length", type=int, default=1024)
parser.add_argument("--global-batch", type=int, default=8)
parser.add_argument(
    "--scope", nargs="+", default=["attn", "moe_router", "moe_preprocess"]
)
parser.add_argument("--skip-checkpoint", action="store_true")
parser.add_argument("--profile", action="store_true")
parser.add_argument("--dropout", type=float, default=0.1)
options = parser.parse_args()
# Record one real iteration in the fresh captured validation run. This run is
# used for numerical/lifecycle evidence; profiler timing is never a benchmark.
if options.graphs and not options.load:
    options.profile = True
root = Path("/artifacts/production")
source = Path("/workspace/Megatron-LM")
out = root / options.name
out.mkdir(exist_ok=False)
args = {
    "--bf16": True,
    "--tensor-model-parallel-size": 1,
    "--pipeline-model-parallel-size": 1,
    "--context-parallel-size": 1,
    "--expert-model-parallel-size": 2,
    "--expert-tensor-parallel-size": 1,
    "--use-megatron-fsdp": True,
    "--megatron-fsdp-version": 2,
    "--data-parallel-sharding-strategy": "optim_grads_params",
    "--megatron-fsdp-main-grads-dtype": "fp32",
    "--overlap-grad-reduce": True,
    "--overlap-param-gather": True,
    "--ckpt-format": "fsdp_dtensor",
    "--no-gradient-accumulation-fusion": True,
    "--deterministic-mode": True,
    "--use-mcore-models": True,
    "--transformer-impl": "transformer_engine",
    "--attention-backend": "unfused",
    "--hybrid-layer-pattern": "*E" * 16,
    "--spec": ["olmoe_spec", "olmoe_stack_spec"],
    "--hidden-size": 2048,
    "--ffn-hidden-size": 1024,
    "--num-attention-heads": 16,
    "--num-query-groups": 16,
    "--kv-channels": 128,
    "--normalization": "RMSNorm",
    "--norm-epsilon": 1e-5,
    "--qk-layernorm": True,
    "--disable-bias-linear": True,
    "--swiglu": True,
    "--untie-embeddings-and-output-weights": True,
    "--position-embedding-type": "rope",
    "--rotary-base": 10000,
    "--max-position-embeddings": 4096,
    "--no-rope-fusion": True,
    "--num-experts": 64,
    "--moe-ffn-hidden-size": 1024,
    "--moe-router-topk": 8,
    "--moe-router-pre-softmax": True,
    "--moe-router-score-function": "softmax",
    "--moe-router-load-balancing-type": "aux_loss",
    "--moe-aux-loss-coeff": 0.01,
    "--moe-router-dtype": "fp32",
    "--moe-grouped-gemm": True,
    "--moe-token-dispatcher-type": "alltoall",
    "--moe-permute-fusion": True,
    "--attention-dropout": options.dropout,
    "--hidden-dropout": options.dropout,
    "--init-method-std": 0.02,
    "--seq-length": options.sequence_length,
    "--micro-batch-size": 1,
    "--global-batch-size": options.global_batch,
    "--train-iters": options.steps,
    "--lr-decay-iters": 1000,
    "--lr-warmup-iters": 20,
    "--lr": 1e-4,
    "--min-lr": 1e-5,
    "--lr-decay-style": "cosine",
    "--adam-beta1": 0.9,
    "--adam-beta2": 0.95,
    "--weight-decay": 0.1,
    "--clip-grad": 1.0,
    "--seed": 1234,
    "--te-rng-tracker": True,
    "--tokenizer-type": "HuggingFaceTokenizer",
    "--tokenizer-model": str(root / "model"),
    "--make-vocab-size-divisible-by": 64,
    "--train-data-path": str(root / "data/wikitext103-train_text_document"),
    "--valid-data-path": str(root / "data/wikitext103-validation_text_document"),
    "--data-cache-path": str(root / "data/cache"),
    "--num-workers": 2,
    "--eval-iters": options.eval_iters,
    "--eval-interval": options.eval_interval,
    "--save": str(out / "checkpoints"),
    "--save-interval": options.save_interval,
    "--tensorboard-dir": str(out / "tensorboard"),
    "--log-interval": 1,
    "--log-memory-interval": 50,
    "--log-timers-to-tensorboard": True,
    "--log-memory-to-tensorboard": True,
    "--log-params-norm": True,
    "--log-validation-ppl-to-tensorboard": True,
    "--log-throughput": True,
    "--distributed-timeout-minutes": 10,
    "--manual-gc": True,
    "--manual-gc-interval": 100,
}
if options.graphs:
    args.update(
        {
            "--cuda-graph-impl": "transformer_engine",
            "--cuda-graph-modules": options.scope,
            "--cuda-graph-warmup-steps": 3,
        }
    )
if options.exit_step:
    args["--exit-interval"] = options.exit_step
if options.load:
    args["--load"] = str(root / options.load / "checkpoints")
    # TE restores optimizer tensors into newly allocated state. Release unused
    # load/step cache before the first lazily initialized NCCL metric communicator.
    # Use the stock training policy; model dimensions and optimizer state are intact.
    args["--empty-unused-memory-level"] = 2
if options.ckpt_step:
    args["--ckpt-step"] = options.ckpt_step
if options.skip_checkpoint:
    args.pop("--save")
    args.pop("--save-interval")
if options.profile:
    args.update(
        {
            "--profile": True,
            "--use-pytorch-profiler": True,
            "--profile-step-start": 10,
            "--profile-step-end": 11,
            "--profile-ranks": [0],
        }
    )
command = [
    sys.executable,
    "-m",
    "torch.distributed.run",
    "--master-addr",
    "127.0.0.1",
    "--master-port",
    "29674",
    "--nproc-per-node",
    "2",
    str(source / "pretrain_hybrid.py"),
]
for key, value in args.items():
    if value is True:
        command.append(key)
    elif isinstance(value, list):
        command.extend([key, *map(str, value)])
    else:
        command.extend([key, str(value)])
env = dict(os.environ)
env.update(
    {
        "CUDA_VISIBLE_DEVICES": "1,3",
        "PYTHONPATH": "/artifacts/production:/workspace/Megatron-LM",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "NCCL_DEBUG": "WARN",
        "NCCL_NVLS_ENABLE": "0",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "OMP_NUM_THREADS": "8",
    }
)
(out / "command.json").write_text(json.dumps(command, indent=2) + "\n")
files = [
    source / relative
    for relative in (
        "megatron/core/transformer/cuda_graphs.py",
        "megatron/core/transformer/module.py",
        "megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/module.py",
        "megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/parameter_group.py",
        "megatron/training/checkpointing.py",
    )
]
files += [Path(__file__), Path(__file__).with_name("olmoe_spec.py")]
(out / "source-sha256.json").write_text(
    json.dumps(
        {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, indent=2
    )
    + "\n"
)
start = time.time()
with (out / "train.log").open("w") as log:
    process = subprocess.run(
        command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT
    )
(out / "exit_code.json").write_text(
    json.dumps({"exit_code": process.returncode, "wall_seconds": time.time() - start})
    + "\n"
)
raise SystemExit(process.returncode)
