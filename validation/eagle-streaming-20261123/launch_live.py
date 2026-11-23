import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

here = Path(__file__).resolve().parent
repo = Path(os.environ.get('EAGLE_REPO_ROOT', '/data/songbinbin/Proj/Proj_LWX/Megatron-LM-eagle-streaming'))
container = 'eagle-streaming-20261123'
uuids = subprocess.check_output(
    ['nvidia-smi', '-i', '1,2', '--query-gpu=uuid', '--format=csv,noheader'], text=True
).splitlines()
endpoint = 'http://127.0.0.1:18876'
deadline = time.monotonic() + 120
while time.monotonic() < deadline:
    ready = subprocess.run(
        ['docker', 'exec', container, 'python', '-c',
         f'import urllib.request; urllib.request.urlopen("{endpoint}/health", timeout=2)'],
        capture_output=True,
    )
    if ready.returncode == 0:
        break
    time.sleep(2)
else:
    raise SystemExit('producer health endpoint did not become ready')
for ranks in (1, 2):
    command = [
        'docker','exec','-e','CUDA_VISIBLE_DEVICES='+','.join(uuids[:ranks]),
        '-e','LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6',
        '-e','PYTHONPATH='+':'.join(str(p) for p in [
            here/'modelopt-install',here/'cpu-deps',here/'shared-python',
            here/'nixl-deps',repo/'examples/post_training/modelopt',repo,
        ]),
        container,'env','-u','NVTE_FLASH_ATTN','-u','NVTE_FUSED_ATTN',
        'timeout','--kill-after=10','150','bash',str(here/'train-live.sh'),
        str(ranks),endpoint,str(here/f'tiny-tp{ranks}-saved'),
    ]
    (here/f'live-tp{ranks}-command.sh').write_text(shlex.join(command)+'\n')
    with (here/f'live-tp{ranks}.log').open('w') as log:
        result = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    print(f'TP{ranks} exit={result.returncode}',flush=True)
    result.check_returncode()
