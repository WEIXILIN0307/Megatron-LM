import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("name")
p.add_argument("--container", default="dcp-stats-training")
p.add_argument("--tp", type=int, default=1)
p.add_argument("--pp", type=int, default=1)
p.add_argument("--cp", type=int)
p.add_argument("--log-interval", type=int, default=1)
p.add_argument("--lengths-file", default="lengths.csv")
p.add_argument("--vpp", action="store_true")
p.add_argument("--on", action="store_true")
p.add_argument("--fixed", action="store_true")
p.add_argument("--audit", action="store_true")
p.add_argument("--measure", action="store_true")
p.add_argument("--no-updates", action="store_true")
p.add_argument("--iters", default="6")
p.add_argument("--load")
p.add_argument("--save", action="store_true")
p.add_argument("--eval", action="store_true")
p.add_argument("--gpus", type=int, default=2)
a = p.parse_args()
root = Path(os.environ.get("DCP_REPO_ROOT", str(Path(__file__).resolve().parents[2])))
base = Path(__file__).resolve().parent
out = base / a.name
out.mkdir(parents=True, exist_ok=True)
cp = a.cp or a.gpus // a.tp // a.pp
assert a.gpus % (a.tp * a.pp * cp) == 0
assert not (a.audit and a.measure)
args = [
    "python",
    "-m",
    "torch.distributed.run",
    "--standalone",
    f"--nproc-per-node={a.gpus}",
    (
        str(base / "train_no_updates.py")
        if a.no_updates
        else (
            str(base / "train_audit.py")
            if a.audit
            else (str(base / "train_measure.py") if a.measure else "pretrain_gpt.py")
        )
    ),
    "--use-mcore-models",
    "--num-layers",
    "4",
    "--hidden-size",
    "128",
    "--ffn-hidden-size",
    "512",
    "--num-attention-heads",
    "4",
    "--seq-length",
    "1024",
    "--max-position-embeddings",
    "1024",
    "--attention-dropout",
    "0",
    "--hidden-dropout",
    "0",
    "--transformer-impl",
    "transformer_engine",
    "--attention-backend",
    "flash",
    "--micro-batch-size",
    "1",
    "--global-batch-size",
    "8",
    "--train-iters",
    a.iters,
    "--lr-decay-iters",
    "100",
    "--lr",
    "0.00015",
    "--min-lr",
    "0.00001",
    "--lr-decay-style",
    "cosine",
    "--lr-warmup-iters",
    "0",
    "--weight-decay",
    "0.01",
    "--clip-grad",
    "1",
    "--bf16",
    "--calculate-per-token-loss",
    "--no-gradient-accumulation-fusion",
    "--tensor-model-parallel-size",
    str(a.tp),
    "--pipeline-model-parallel-size",
    str(a.pp),
    "--context-parallel-size",
    str(cp),
    "--max-seqlen-per-dp-cp-rank",
    "512",
    "--use-varlen-dataset",
    "--mock-data",
    "--varlen-mock-dataset-config-json",
    json.dumps({"mode": "file", "format": "thd", "path": str(base / a.lengths_file)}),
    "--tokenizer-type",
    "NullTokenizer",
    "--vocab-size",
    "256",
    "--null-tokenizer-pad-id",
    "0",
    "--split",
    "99,1,0",
    "--data-cache-path",
    str(base / "data-cache"),
    "--dataloader-type",
    "single",
    "--num-workers",
    "0",
    "--log-interval",
    str(a.log_interval),
    "--log-throughput",
    "--timing-log-level",
    "0",
    "--eval-interval",
    "3" if a.eval else "1000000",
    "--eval-iters",
    "1" if a.eval else "0",
    "--save-interval",
    a.iters,
    "--distributed-backend",
    "nccl",
    "--distributed-timeout-minutes",
    "5",
    "--seed",
    "1234",
    "--no-one-logger",
    "--tensorboard-dir",
    str(out / "tensorboard"),
]
args += (
    ["--sequence-packing-scheduler", "dp_balanced"]
    if a.fixed
    else [
        "--dynamic-context-parallel",
        "--sequence-packing-scheduler",
        "default_dynamic_cp",
    ]
)
if a.tp > 1:
    args += ["--sequence-parallel"]
if a.vpp:
    args += [
        "--num-layers-per-virtual-pipeline-stage",
        "1",
        "--microbatch-group-size-per-virtual-pipeline-stage",
        "2",
    ]
if a.on:
    args += ["--log-dcp-stats"]
if a.no_updates:
    args += ["--no-load-optim"]
if a.load:
    args += [
        "--use-checkpoint-opt-param-scheduler",
        "--exit-on-missing-checkpoint",
        "--load",
        str(base / a.load / "checkpoints"),
    ]
if a.save:
    args += ["--save", str(out / "checkpoints"), "--ckpt-format", "torch"]
uuids = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
).splitlines()
profile_env = ["-e", "DCP_PROFILE_CPU=1"] if os.environ.get("DCP_PROFILE_CPU") else []
binding_env = (
    ["-e", "DCP_MEASURE_CPU_BINDINGS=" + os.environ["DCP_MEASURE_CPU_BINDINGS"]]
    if os.environ.get("DCP_MEASURE_CPU_BINDINGS")
    else []
)
cmd = [
    "docker",
    "exec",
    *binding_env,
    *profile_env,
    "-e",
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO=0",
    "-e",
    "CUBLAS_WORKSPACE_CONFIG=:4096:8",
    "-e",
    "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1",
    "-e",
    "CUDA_VISIBLE_DEVICES="
    + ",".join(
        uuids[i]
        for i in (
            list(map(int, os.environ["DCP_GPU_INDICES"].split(",")))
            if os.environ.get("DCP_GPU_INDICES")
            else ((4, 5) if a.gpus == 2 else (1, 2, 4, 5))
        )
    ),
    "-e",
    "NVTE_FLASH_ATTN=1",
    "-e",
    "NVTE_FUSED_ATTN=0",
    "-e",
    "NVTE_DEBUG=0",
    "-e",
    "NVTE_DEBUG_LEVEL=0",
    "-e",
    f"DCP_AUDIT_DIR={out}/audit",
    "--workdir",
    str(root),
    a.container,
] + args
(out / "command.sh").write_text(shlex.join(cmd) + "\n")
with (out / "train.log").open("w") as f:
    r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
print(a.name, r.returncode, flush=True)
raise SystemExit(r.returncode)
