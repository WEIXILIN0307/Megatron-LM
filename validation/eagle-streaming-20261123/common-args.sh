teacher="/data/songbinbin/Proj/Proj_LWX/.codex-artifacts/eagle-streaming-20261123/tiny-qwen3"
common=(
  --num-layers 4 --hidden-size 128 --ffn-hidden-size 384
  --num-attention-heads 4 --group-query-attention --num-query-groups 2 --kv-channels 32
  --normalization RMSNorm --norm-epsilon 1e-6 --position-embedding-type rope
  --rotary-base 10000 --swiglu --disable-bias-linear --make-vocab-size-divisible-by 128
  --seq-length 64 --max-position-embeddings 256 --bf16
  --tokenizer-type HuggingFaceTokenizer --tokenizer-model "$teacher"
  --export-offline-model --export-model-type GPTModel --ckpt-format torch_dist
  --no-gradient-accumulation-fusion --no-masked-softmax-fusion --no-bias-swiglu-fusion
  --no-one-logger --attention-dropout 0 --hidden-dropout 0
)
