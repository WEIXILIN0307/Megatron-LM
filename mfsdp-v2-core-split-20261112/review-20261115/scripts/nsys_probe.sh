#!/bin/bash
set -euo pipefail
exec /usr/local/cuda/bin/nsys profile \
  --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node \
  --cuda-event-trace=false --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output="${PROOF_OUTPUT}/trace-rank-${RANK}" \
  /home/songbinbin/.venvs/megatron-gdn-offload/bin/python \
  /data/songbinbin/Proj/Proj_LWX/.codex-artifacts/mfsdp-review-20261115/current_trace_probe.py
