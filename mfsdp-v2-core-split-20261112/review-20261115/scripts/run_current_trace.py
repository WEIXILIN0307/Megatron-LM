"""Record a bounded trace on the exact published source; no timing claim."""

import hashlib
import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parent
source = root / "core"
output = Path("/data01/LWX/mfsdp-review-20261115/trace-pinned-head")
output.mkdir(exist_ok=False)
command = json.loads(Path("/data01/LWX/mfsdp-review-20261115/trace-current-ep2-edp2-fixed/command.json").read_text())
command = [arg.replace("trace-current-ep2-edp2-fixed", output.name) for arg in command]
(output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
status = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True)
assert commit == "e497dd0a2247e13bc9ed5c974a0bb286f4f599a4" and not status
(output / "source.json").write_text(json.dumps({"commit": commit, "status": status,
    "probe_sha256": hashlib.sha256((root / "current_trace_probe.py").read_bytes()).hexdigest(),
    "tracer_sha256": hashlib.sha256((root / "trace_units.py").read_bytes()).hexdigest()}, indent=2) + "\n")
(output / "preflight.json").write_text(json.dumps({"gpus": subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,uuid,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True),
    "note": "Focused correctness/concurrency only; shared host, not a performance benchmark."}, indent=2) + "\n")
with (output / "launcher.log").open("w") as log:
    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
(output / "exit.json").write_text(json.dumps({"exit_code": result.returncode,
    "options": {"cuda_event_trace": "false"}}, indent=2) + "\n")
print("Pinned current-head trace exit", result.returncode, flush=True)
if result.returncode:
    raise SystemExit(result.returncode)
for rank in range(4):
    base = "/proof/" + output.name + f"/trace-rank-{rank}"
    subprocess.run(["docker", "exec", "mfsdp-review-20261115", "/usr/local/cuda/bin/nsys", "export",
                    "--type=sqlite", "--output=" + base + ".sqlite", base + ".nsys-rep"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
analyzer = root.parent / "mfsdp-gap-closure-20261113/analyze_trace.py"
subprocess.run(["python", "-S", str(analyzer), str(output)], check=True, stdout=subprocess.DEVNULL)
print("Exported and analyzed all four native traces.", flush=True)
