#!/bin/bash
set -euo pipefail
exec /usr/local/cuda/bin/nsys profile \
  --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node --cuda-event-trace="${PROOF_CUDA_EVENT_TRACE}" \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output="${PROOF_OUTPUT}/trace-rank-${RANK}" \
  /home/songbinbin/.venvs/megatron-gdn-offload/bin/python \
  /data/songbinbin/Proj/Proj_LWX/.codex-artifacts/mfsdp-gap-closure-20261113/profile_training.py "$@"
