set -e
exec docker exec \
  -e LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6:/home/songbinbin/.cache/uv/archive-v0/uf1UUojg_3oGCYLI/nvidia/nccl/lib/libnccl.so.2 \
  -e LD_LIBRARY_PATH=/opt/nvidia:/home/songbinbin/vllm/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:/home/songbinbin/.venvs/megatron-gdn-offload/lib/python3.12/site-packages/nvidia/cu13/lib \
  -e PYTHONPATH=/data/songbinbin/Proj/Proj_LWX/.codex-artifacts/eagle-streaming-20261123/producer-env:/data/songbinbin/Proj/Proj_LWX/.codex-artifacts/eagle-streaming-20261123/modelopt-install:/data/songbinbin/Proj/Proj_LWX/.codex-artifacts/eagle-streaming-20261123/nixl-deps:/data/songbinbin/Proj/Proj_LWX/.codex-artifacts/eagle-streaming-20261123/cpu-deps \
  -e CUDA_VISIBLE_DEVICES="$(nvidia-smi -i 5 --query-gpu=uuid --format=csv,noheader)" \
  -e VLLM_USE_FLASHINFER_SAMPLER=0 eagle-streaming-producer-cu13-20261123 bash /data/songbinbin/Proj/Proj_LWX/.codex-artifacts/eagle-streaming-20261123/serve-tiny.sh
