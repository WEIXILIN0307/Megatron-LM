"""Apply the two reviewed Python hunks to an isolated installed-package overlay."""

import hashlib
import json
import subprocess
import sys
import textwrap
from pathlib import Path

task = Path("/data01/LWX/mfsdp-gap-closure-20261113")
repo = task / "TransformerEngine"
original = Path("/home/songbinbin/.venvs/megatron-gdn-offload/lib/python3.12/site-packages/transformer_engine")
baseline = subprocess.check_output(["git", "-C", str(repo), "show", "HEAD:transformer_engine/pytorch/graph.py"], text=True)
candidate = (repo / "transformer_engine/pytorch/graph.py").read_text()
before = (original / "pytorch/graph.py").read_text()
old_hooks = """        hooks = []
        for module in func.modules():
            hooks.append(module.register_forward_hook(hook_fn))
        outputs = func(*args, **kwargs)
        for hook in hooks:
            hook.remove()
"""
new_hooks = """        hooks = []
        try:
            for module in func.modules():
                hooks.append(module.register_forward_hook(hook_fn))
            outputs = func(*args, **kwargs)
        finally:
            for hook in hooks:
                hook.remove()
"""
start = before.index("    graphed_callables = _make_graphed_callables(")
end = before.index("    return graphed_callables", start)
old_outer = before[start:end]
boundary = old_outer.index("    # Ensures warmup does not affect numerics")
# Keep the installed release's graph-safe RNG compatibility branch unchanged.
new_outer = ("    try:\n" + textwrap.indent(old_outer[:boundary], "    ")
             + "    finally:\n" + textwrap.indent(old_outer[boundary:], "    "))
assert before.count(old_hooks) == before.count(old_outer) == 1
after = before.replace(old_hooks, new_hooks).replace(old_outer, new_outer)
main_source = "--upstream-graph" in sys.argv
if main_source:
    after = candidate
overlay_name = "te-main-overlay" if main_source else "te-overlay"
overlay = task / overlay_name / "transformer_engine"
overlay.mkdir(parents=True, exist_ok=False)
for item in original.iterdir():
    if item.name == "__pycache__":
        continue
    if item.name == "pytorch":
        (overlay / item.name).mkdir()
        for child in item.iterdir():
            if child.name not in ("graph.py", "__pycache__"):
                (overlay / item.name / child.name).symlink_to(child, target_is_directory=child.is_dir())
    else:
        (overlay / item.name).symlink_to(item, target_is_directory=item.is_dir())
(overlay / "pytorch/graph.py").write_text(after)
(task / (overlay_name + "-manifest.json")).write_text(json.dumps({
    "mode": "upstream graph.py" if main_source else "installed release with equivalent cleanup hunks",
    "source_head": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
    "installed_sha256": hashlib.sha256(before.encode()).hexdigest(),
    "upstream_baseline_sha256": hashlib.sha256(baseline.encode()).hexdigest(),
    "overlay_sha256": hashlib.sha256(after.encode()).hexdigest(),
    "installed_package_modified": False,
    "native_extension_modified": False,
    "changes": ["finally remove temporary warmup hooks", "finally restore capture-owned RNG, dispatch, FP8 state and flag"],
}, indent=2) + "\n")
