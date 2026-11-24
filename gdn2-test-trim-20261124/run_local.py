"""Record bounded GPU tests through the existing project container."""

import json
import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).parent
name, source, devices, *selection = sys.argv[1:]
output = root / "results" / name
output.mkdir(parents=True, exist_ok=False)
rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.free",
                               "--format=csv,noheader,nounits"], text=True)
available = {r[0]: r for r in (line.split(", ") for line in rows.splitlines())}
selected = [available[i] for i in devices.split(",")]
assert all(int(r[2]) > 12000 for r in selected), selected
(output / "preflight.json").write_text(json.dumps(selected, indent=2) + "\n")
(output / "source.json").write_text(json.dumps({
    "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root / source, text=True).strip(),
    "status": subprocess.check_output(["git", "status", "--porcelain"], cwd=root / source, text=True),
}, indent=2) + "\n")
command = ["docker", "exec", "--workdir", str(root / source)]
for key, value in {
    "CUDA_VISIBLE_DEVICES": ",".join(r[1] for r in selected),
    "PYTHONPATH": f"{root / source}:{root / 'runtime'}:/existing-artifacts/causal-conv1d-python:/existing-artifacts/flash-attn-python",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "NVTE_CPU_OFFLOAD_V1": "1",
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
    "OMP_NUM_THREADS": "1",
    "TRITON_CACHE_DIR": str(root / "cache" / "triton"),
    "TORCHINDUCTOR_CACHE_DIR": str(root / "cache" / "inductor"),
}.items():
    command += ["--env", f"{key}={value}"]
command += [
    "mfsdp-partial-cg-20261107", "timeout", "900",
    "/home/songbinbin/.venvs/megatron-gdn-offload/bin/python",
    "-m", "torch.distributed.run", "--standalone",
    f"--nproc-per-node={len(devices.split(','))}",
    "--log-dir", str(output / "ranks"), "--redirects", "3",
    str(root / "runtime" / "rank_distributed_pytest.py"), str(output / "reports"),
    "-vv", "-s", "-p", "no:cacheprovider", *selection,
]
(output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
with (output / "launcher.log").open("w") as log:
    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
(output / "exit.json").write_text(json.dumps({"exit_code": result.returncode}) + "\n")
print(name, result.returncode)
for report in sorted((output / "reports").glob("*.json")):
    data = json.loads(report.read_text())
    print(report.name, data["exit_code"], [(x["nodeid"], x["outcome"]) for x in data["outcomes"]])
sys.exit(result.returncode)
