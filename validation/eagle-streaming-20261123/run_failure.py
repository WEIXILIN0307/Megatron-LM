import http.server
import os
import subprocess
import sys
import threading
import time
from pathlib import Path


class Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        print('source reached HTTP teacher endpoint', flush=True)
        time.sleep(10)


here = Path(__file__).resolve().parent
with http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler) as server:
    threading.Thread(target=server.serve_forever, daemon=True).start()
    started = time.monotonic()
    result = subprocess.run(
        [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=2',
         '--monitor-interval=0.2', str(here / 'probe_tp_failure.py')],
        env=dict(os.environ, PROBE_SERVER=f'http://127.0.0.1:{server.server_port}',
                 PROBE_TOKENIZER=str(here / 'qwen3-teacher')),
        timeout=60,
    )
    print(f'torchrun exit={result.returncode} elapsed={time.monotonic()-started:.1f}s', flush=True)
    assert result.returncode == 1
