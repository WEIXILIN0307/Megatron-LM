"""Run a recorded small GPU regression on A6000PRO without stopping other jobs."""

import json
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

root = Path(__file__).parent
name, source, devices, *selection = sys.argv[1:]
base = "/home/lwx/.cache/megatron-gdn-20261026"
remote_source = f"{base}/gdn2-test-observer-20261124-{source}"
output = f"{base}/gdn2-test-observer-20261124/{name}"
subprocess.run(["rsync", "-a", "--exclude=.git", "--exclude=__pycache__",
                str(root / source) + "/", f"A6000PRO:{remote_source}/"], check=True)
query = subprocess.check_output(["ssh", "A6000PRO", "nvidia-smi", "-q", "-x"], text=True)
rows = {str(i): [str(i), gpu.findtext("minor_number"), gpu.findtext("uuid"),
                gpu.findtext("fb_memory_usage/free").split()[0]]
        for i, gpu in enumerate(ET.fromstring(query).findall("gpu"))}
selected = [rows[i] for i in devices.split(",")]
assert all(int(r[3]) > 12000 for r in selected), selected
wrapper = subprocess.check_output(["ssh", "A6000PRO", "cat",
    base + "/precision_graph_container_gpu1_default.py"], text=True)
wrapper = wrapper.replace("('nvidia1', 'nvidiactl'", "(" + ", ".join(repr("nvidia" + r[1]) for r in selected) + ", 'nvidiactl'")
import re
wrapper = re.sub(r"CUDA_VISIBLE_DEVICES='[^']+'", "CUDA_VISIBLE_DEVICES=" + repr(",".join(r[2] for r in selected)), wrapper)
wrapper = wrapper.replace("NVTE_NVFP4_DISABLE_RHT='0'", "NVTE_NVFP4_DISABLE_RHT='1'").replace("NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING='0'", "NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING='1'")
local = root / "runtime" / f"{name}-container.py"
local.write_text(wrapper)
subprocess.run(["ssh", "A6000PRO", "mkdir", "-p", output], check=True)
subprocess.run(["scp", str(local), f"A6000PRO:{output}/container.py"], check=True)
environment = {
    "GDN_SOURCE": remote_source, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    "GDN_TRITON_CACHE_DIR": f"{base}/scope-20261123/cache/triton",
    "GDN_INDUCTOR_CACHE_DIR": f"{base}/scope-20261123/cache/inductor",
}
command = ["env", *(f"{k}={v}" for k, v in environment.items()), "timeout", "900",
    "/usr/bin/python3", output + "/container.py",
    "/home/songbinbin/.venvs/megatron-gdn-offload/bin/python", "-m", "torch.distributed.run",
    "--standalone", f"--nproc-per-node={len(selected)}", "--log-dir", output + "/ranks", "--redirects", "3",
    base + "/rank_env_precision_graph.py", base + "/rank_distributed_pytest.py", output + "/reports",
    "-vv", "-s", "-p", "no:cacheprovider", *selection]
local_output = root / "results" / name
local_output.mkdir(parents=True, exist_ok=False)
(local_output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
(local_output / "preflight.json").write_text(json.dumps(selected, indent=2) + "\n")
(local_output / "source.json").write_text(json.dumps({
    "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root / source, text=True).strip(),
    "status": subprocess.check_output(["git", "status", "--porcelain"], cwd=root / source, text=True),
}, indent=2) + "\n")
result = subprocess.run(["ssh", "A6000PRO", "unset NVTE_BACKWARD_OVERRIDE; " + shlex.join(command) + " > " + shlex.quote(output + "/launcher.log") + " 2>&1"])
subprocess.run(["rsync", "-a", f"A6000PRO:{output}/", str(local_output) + "/"], check=True)
(local_output / "exit.json").write_text(json.dumps({"exit_code": result.returncode}) + "\n")
print(name, result.returncode, flush=True)
sys.exit(result.returncode)
