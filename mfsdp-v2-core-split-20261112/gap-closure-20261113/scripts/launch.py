"""Run one bounded, reproducible proxy experiment in the existing local image."""

import argparse
import datetime
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("name")
parser.add_argument("--mode", choices=["eager", "graph", "serialized", "empty-stream"], required=True)
parser.add_argument("--gpus", default="0,1,2")
parser.add_argument("--ep", type=int, default=1)
parser.add_argument("--steps", type=int, default=45)
parser.add_argument("--first", type=int, default=16)
parser.add_argument("--last", type=int, default=45)
special = parser.add_mutually_exclusive_group()
special.add_argument("--trace", action="store_true")
special.add_argument("--state-digests", action="store_true")
parser.add_argument("--cuda-event-trace", choices=["true", "false"], default="true")
parser.add_argument("--capture-provenance", action="store_true")
parser.add_argument("--gradient-focus", action="store_true")
options = parser.parse_args()
if options.state_digests and options.mode in ("serialized", "empty-stream"):
    parser.error("State digests compare ordinary eager and graph training only.")
root = Path(__file__).resolve().parent
workspace = root.parent.parent
source = workspace / ".codex-artifacts/mfsdp-three-pr-split-20261112/composed"
host_root = Path("/data01/LWX/mfsdp-gap-closure-20261113")
output = host_root / "runs" / options.name
output.mkdir(parents=True, exist_ok=False)
gpu_rows = subprocess.check_output([
    "nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu",
    "--format=csv,noheader,nounits"], text=True)
rows = {int(r[0]): r for r in [line.split(", ") for line in gpu_rows.strip().splitlines()]}
indices = [int(i) for i in options.gpus.split(",")]
assert len(set(indices)) == len(indices)
assert len(indices) % options.ep == 0
assert all(int(rows[i][4]) > 22 * 1024 for i in indices), gpu_rows
uuids = [rows[i][1] for i in indices]
(output / "preflight-gpu.csv").write_text(gpu_rows)
(output / "topology.txt").write_text(subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True))
args = json.loads((workspace / ".codex-artifacts/mfsdp-rfc-design-20261112/model-args.json").read_text())
changes = {
    "--expert-model-parallel-size": str(options.ep),
    "--global-batch-size": str(2 * len(indices)),
    "--train-iters": str(options.steps),
    "--log-interval": "100000",
    "--tokenizer-model": "/proof/model",
    "--train-data-path": "/proof/data/wikitext103-train_text_document",
    "--valid-data-path": "/proof/data/wikitext103-validation_text_document",
    "--data-cache-path": "/proof/data/cache",
}
for key, value in changes.items():
    args[args.index(key) + 1] = value
if options.mode != "eager":
    args += ["--cuda-graph-impl", "transformer_engine", "--cuda-graph-modules",
             "attn", "moe_router", "moe_preprocess", "--cuda-graph-warmup-steps", "3"]
env = {
    "CUDA_VISIBLE_DEVICES": ",".join(uuids),
    "PATH": "/home/songbinbin/.venvs/megatron-gdn-offload/bin:/data/songbinbin/minicoda/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
    "PYTHONPATH": str(root) + ":" + str(source),
    "PROOF_OUTPUT": "/proof/runs/" + options.name,
    "PROOF_SOURCE": str(source),
    "PROOF_MODE": options.mode,
    "PROOF_FIRST": str(options.first),
    "PROOF_LAST": str(options.last),
    "PROOF_CUDA_EVENT_TRACE": options.cuda_event_trace,
    "PROOF_CAPTURE_PROVENANCE": str(int(options.capture_provenance)),
    "PROOF_GRADIENT_FOCUS": str(int(options.gradient_focus)),
    "FOLLOWUP_OUTPUT": "/proof/runs/" + options.name,
    "FOLLOWUP_SOURCE": str(source),
    "FOLLOWUP_VARIANT": options.name,
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
    "TMPDIR": "/proof/tmp",
}
(host_root / "tmp").mkdir(exist_ok=True)
command = ["docker", "exec", "--workdir", str(source)]
for key, value in env.items():
    command += ["--env", key + "=" + value]
command += ["mfsdp-gap-closure-20261113", "/usr/bin/timeout", "--signal=TERM", "--kill-after=15s", "900",
            "/home/songbinbin/.venvs/megatron-gdn-offload/bin/python", "-m", "torch.distributed.run",
            "--nnodes", "1", "--master-addr", "127.0.0.1", "--master-port", "29879",
            "--nproc-per-node", str(len(indices)), "--log-dir", "/proof/runs/" + options.name + "/torchrun",
            "--redirects", "3"]
if options.trace:
    command += ["--no-python", "/bin/bash", str(root / "nsys_worker.sh")]
elif options.state_digests:
    command += [str(root / "verify_training.py")]
else:
    command += [str(root / "measure_training.py")]
command += args
(output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
stop = threading.Event()


def monitor():
    with (output / "gpu-telemetry.jsonl").open("w") as log:
        while not stop.is_set():
            gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.free,power.draw,clocks.sm,temperature.gpu",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True)
            processes = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                                        "--format=csv,noheader,nounits"], capture_output=True, text=True)
            log.write(json.dumps({"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                  "gpu_csv": gpu.stdout, "process_csv": processes.stdout,
                                  "loadavg": os.getloadavg()}) + "\n")
            log.flush()
            stop.wait(3)


thread = threading.Thread(target=monitor, daemon=True)
thread.start()
start = datetime.datetime.now(datetime.timezone.utc).isoformat()
try:
    with (output / "launcher.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
finally:
    stop.set()
    thread.join(timeout=10)
status = {"exit_code": result.returncode, "start_utc": start,
          "end_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "options": vars(options)}
(output / "exit.json").write_text(json.dumps(status, indent=2) + "\n")
print(json.dumps(status), flush=True)
sys.exit(result.returncode)
