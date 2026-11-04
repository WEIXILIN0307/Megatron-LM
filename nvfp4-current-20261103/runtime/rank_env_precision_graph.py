"""Give each rank sole ownership of both persistent compiler caches."""

import os
import sys
from pathlib import Path

if __name__ == "__main__":
    for name in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
        os.environ[name] = str(Path(os.environ[name]) / f"rank-{os.environ['RANK']}")
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
