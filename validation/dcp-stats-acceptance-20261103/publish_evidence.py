"""Copy only explicit acceptance records into the existing evidence worktree."""
import hashlib
import json
import shutil
from pathlib import Path

src = Path(__file__).resolve().parent
dst = Path('/data/songbinbin/Proj/Proj_LWX/Megatron-LM-dcp-rebase-evidence-20261103/validation/dcp-stats-acceptance-20261103')
dst.mkdir(parents=True, exist_ok=True)
files = [
    'README.md', 'source.json', 'tested-file-hashes.json', 'checkpoint-hashes.json',
    'environment.json', 'container-image.txt', 'container-command.txt',
    'gpu-topology.txt', 'cpu-topology.txt', 'gpu-snapshot.txt',
    'launch_train.py', 'train_audit.py', 'train_measure.py', 'train_no_updates.py',
    'run_performance.py', 'analyze_performance.py', 'verify_performance.py',
    'run_diagnostics.py', 'analyze_cpu_profile.py', 'run_dp2.py', 'verify_dp2.py',
    'verify_no_updates.py', 'lengths.csv', 'dp2-lengths.csv',
    'current-gpu.csv', 'current-performance-runs.json', 'current-performance.json',
    'cpu-profile-summary.json', 'performance-checks.json', 'dp2-correctness.json',
    'no-updates-check.json', 'analyze-performance.log', 'analyze-cpu-profile.log',
    'verify_performance.log', 'verify_dp2.log', 'verify_no_updates.log',
    'unit-command.txt', 'pytest.log', 'isort-evidence.log', 'black-evidence.log',
    'publish_evidence.py',
]
for name in files:
    shutil.copy2(src / name, dst / name)
run_names = [row['name'] for row in json.loads((src / 'current-performance-runs.json').read_text())]
run_names += ['initial-aligned', 'initial-dp2', 'dp2-off', 'dp2-on', 'cpu-profile-off', 'cpu-profile-on', 'no-updates']
for name in run_names:
    for path in (src / name).rglob('*'):
        if not path.is_file():
            continue
        relative = path.relative_to(src / name)
        if 'audit' in relative.parts or 'tensorboard' in relative.parts or path.name in ('command.sh', 'train.log'):
            target = dst / path.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
# Preserve rejected setup runs, but never their checkpoint binaries or caches.
for path in (src / 'setup-attempts').rglob('*'):
    if path.is_file() and 'checkpoints' not in path.parts:
        target = dst / path.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
manifest = ''.join(hashlib.sha256(path.read_bytes()).hexdigest() + '  ' + path.relative_to(dst).as_posix() + '\n' for path in sorted(dst.rglob('*')) if path.is_file() and path.name != 'SHA256SUMS')
(dst / 'SHA256SUMS').write_text(manifest)
print(json.dumps({'files': len(manifest.splitlines()), 'total_bytes': sum(path.stat().st_size for path in dst.rglob('*') if path.is_file()), 'destination': str(dst)}, indent=2))
