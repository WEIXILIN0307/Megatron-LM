set -e
source "/data/songbinbin/Proj/Proj_LWX/.codex-artifacts/eagle-streaming-20261123/common-args.sh"
ranks="$1"
endpoint="$2"
out="$3"
python -m torch.distributed.run --standalone --nproc-per-node="$ranks" --monitor-interval=0.2 \
  examples/post_training/modelopt/finetune.py "${common[@]}" \
  --tensor-model-parallel-size "$ranks" --load "/data/songbinbin/Proj/Proj_LWX/.codex-artifacts/eagle-streaming-20261123/tiny-offline" --exit-on-missing-checkpoint \
  --finetune --no-load-optim --no-load-rng --save "$out" --save-interval 2 --no-save-optim --no-save-rng \
  --micro-batch-size 1 --global-batch-size 1 --train-iters 2 --lr 1e-4 --lr-decay-style constant \
  --mock-data --dataloader-type single --num-workers 0 --eval-iters 1 --eval-interval 1 --log-interval 1 \
  --eagle-streaming-server-urls "$endpoint" --eagle-streaming-model qwen3-teacher \
  --eagle-streaming-train-data "/data/songbinbin/Proj/Proj_LWX/.codex-artifacts/eagle-streaming-20261123/train.jsonl" \
  --eagle-streaming-valid-data "/data/songbinbin/Proj/Proj_LWX/.codex-artifacts/eagle-streaming-20261123/valid.jsonl"
