"""Run bounded pytest probes through the existing GPU container, recording exact commands."""

import argparse
import json
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("name")
p.add_argument("--source", default="core")
p.add_argument("--gpus", default="1,6")
p.add_argument("--selection", default="")
p.add_argument("--te-overlay", action="store_true")
p.add_argument("--file", default="tests/unit_tests/distributed/mfsdp_v2/test_mcore_partial_cuda_graph.py")
args = p.parse_args()
root = Path(__file__).resolve().parent
source = root / args.source
output = root / "validation" / args.name
output.mkdir(parents=True, exist_ok=False)
(output / "tmp").mkdir()
rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.free,utilization.gpu",
                                "--format=csv,noheader,nounits"], text=True)
devices = {r[0]: r for r in (line.split(", ") for line in rows.splitlines())}
indices = args.gpus.split(",")
assert all(int(devices[i][2]) > 16000 for i in indices)
(output / "preflight.json").write_text(json.dumps([devices[i] for i in indices], indent=2) + "\n")
command = ["docker", "exec", "--workdir", str(source)]
environment = {"CUDA_VISIBLE_DEVICES": ",".join(devices[i][1] for i in indices),
               "PYTHONPATH": (str(root / "te-overlay") + ":" if args.te_overlay else "") + str(source),
               "PYTHONDONTWRITEBYTECODE": "1",
               # Let pytest assign one directory per process. A common --basetemp
               # races on creation; the distributed fixture joins their common parent.
               "PYTEST_DEBUG_TEMPROOT": str(output / "tmp"),
               "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0", "OMP_NUM_THREADS": "2"}
for key, value in environment.items():
    command += ["--env", key + "=" + value]
command += ["mfsdp-partial-cg-20261107", "/usr/bin/timeout", "--kill-after=15s", "600",
            "/home/songbinbin/.venvs/megatron-gdn-offload/bin/python", "-m", "torch.distributed.run",
            "--standalone", "--nnodes=1", "--nproc-per-node=" + str(len(indices)),
            "--log-dir=" + str(output / "torchrun"), "--redirects=3", "-m", "pytest", "-q",
            args.file, "--experimental", "--maxfail=1", "--capture=fd", "-o", "addopts="]
if args.selection:
    command += ["-k", args.selection]
(output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
(output / "source.json").write_text(json.dumps({
    "commit": subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
    "status": subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True),
}, indent=2) + "\n")
with (output / "launcher.log").open("w") as log:
    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
(output / "exit.json").write_text(json.dumps({"exit_code": result.returncode, "options": vars(args)}, indent=2) + "\n")
print(args.name, "exit", result.returncode, flush=True)
for path in sorted(output.glob("torchrun/**/stdout.log")):
    summaries = [line for line in path.read_text().splitlines()
                 if " passed" in line or " failed" in line or " skipped" in line or line.startswith("ERROR")]
    print(path.parent.name, "\n".join(summaries[-3:]), flush=True)
raise SystemExit(result.returncode)
