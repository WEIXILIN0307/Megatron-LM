set -e
python - <<'PY'
from pathlib import Path
# The CSV directly specifies valid content-token lengths.
Path('/tmp/dcp-lengths.csv').write_text(('1024\n256\n128\n128\n512\n256\n128\n128\n') * 24)
PY
COMMON=(
  --use-mcore-models --num-layers 4 --hidden-size 128 --ffn-hidden-size 512
  --num-attention-heads 4 --seq-length 1024 --max-position-embeddings 1024
  --transformer-impl transformer_engine --attention-backend flash --bf16
  --attention-dropout 0 --hidden-dropout 0 --no-gradient-accumulation-fusion
  --micro-batch-size 1 --global-batch-size 8 --calculate-per-token-loss
  --context-parallel-size 2 --max-seqlen-per-dp-cp-rank 512
  --use-varlen-dataset --mock-data --tokenizer-type NullTokenizer --vocab-size 256
  --null-tokenizer-pad-id 0 --num-workers 0 --dataloader-type single
  --varlen-mock-dataset-config-json '{"mode":"file","format":"thd","path":"/tmp/dcp-lengths.csv"}'
  --train-iters 5 --lr-decay-iters 100 --lr 0.00015 --min-lr 0.00001
  --lr-warmup-iters 0 --seed 1234 --log-interval 1 --eval-iters 0 --no-one-logger
)
NVTE_FLASH_ATTN=1 NVTE_FUSED_ATTN=0 CUDA_DEVICE_MAX_CONNECTIONS=1 \
  python -m torch.distributed.run --standalone --nproc-per-node 2 pretrain_gpt.py \
  "${COMMON[@]}" --dynamic-context-parallel \
  --sequence-packing-scheduler default_dynamic_cp --log-dcp-stats
