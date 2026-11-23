set -e
base="/data/songbinbin/Proj/Proj_LWX/.codex-artifacts/eagle-streaming-20261123"
repo="/data/songbinbin/Proj/Proj_LWX/Megatron-LM-eagle-streaming"
launch=(
  docker exec
  -e CUDA_VISIBLE_DEVICES="$(nvidia-smi -i 1,2 --query-gpu=uuid --format=csv,noheader | paste -sd,)"
  -e LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
  -e "PYTHONPATH=$base/modelopt-install:$base/cpu-deps:$base/shared-python:$base/nixl-deps:$repo/examples/post_training/modelopt:$repo"
  eagle-streaming-20261123
)
"${launch[@]}" python "$base/run_failure.py" > "$base/tp-failure-final.log" 2>&1
set +e
"${launch[@]}" env -u NVTE_FLASH_ATTN -u NVTE_FUSED_ATTN timeout --kill-after=10 90 \
  bash "$base/train-live.sh" 2 http://127.0.0.1:1 "$base/failed-checkpoint" \
  > "$base/launcher-refused-final.log" 2>&1
status=$?
set -e
printf 'TP2 finetune exit=%s\n' "$status"
test "$status" -eq 1
for needle in 'successfully loaded checkpoint' 'Connection refused' 'no fetchable sample' 'closing signal SIGTERM'; do
  rg -q "$needle" "$base/launcher-refused-final.log"
done
