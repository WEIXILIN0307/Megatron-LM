"""Separate CPU-only diagnostic; excluded from benchmark aggregates."""

import os
import subprocess
import sys
from pathlib import Path

base = Path(__file__).resolve().parent
for arm in ("off", "on"):
    args = [
        sys.executable,
        "-S",
        str(base / "launch_train.py"),
        "cpu-profile-" + arm,
        "--container",
        "dcp-stats-acceptance-20261103",
        "--measure",
        "--iters",
        "41",
        "--load",
        "initial-aligned",
    ]
    if arm == "on":
        args += ["--on"]
    subprocess.run(
        args,
        env=dict(
            os.environ,
            DCP_REPO_ROOT="/data/songbinbin/Proj/Proj_LWX/Megatron-LM-dcp-stats",
            DCP_GPU_INDICES="4,5",
            DCP_PROFILE_CPU="1",
            DCP_MEASURE_CPU_BINDINGS="[[8,9,10,11,40,41,42,43],[12,13,14,15,44,45,46,47]]",
        ),
        check=True,
    )
