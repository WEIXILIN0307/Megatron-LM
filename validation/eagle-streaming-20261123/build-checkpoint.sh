set -e
source "/data/songbinbin/Proj/Proj_LWX/.codex-artifacts/eagle-streaming-20261123/common-args.sh"
python -m torch.distributed.run --standalone --nproc-per-node=1 "/data/songbinbin/Proj/Proj_LWX/.codex-artifacts/eagle-streaming-20261123/make_tiny_checkpoint.py" \
  "${common[@]}" --save "/data/songbinbin/Proj/Proj_LWX/.codex-artifacts/eagle-streaming-20261123/tiny-offline" --save-interval 1 --no-save-optim --no-save-rng \
  --micro-batch-size 1 --global-batch-size 1 --train-iters 1 --eval-iters 0 --mock-data
