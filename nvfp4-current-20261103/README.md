# NVFP4 validation on RTX PRO 6000 SM120

#8543 source: `81b5e39259bbebbcc951d1e5e269ed3ce190bde5`, including NVIDIA main at `d00e888f270e2270da872a9679d490f9411e4ece`.

#8556 source: `66b6bb62a00bdccf8365096e6828215cffed4ce7`. Its regression control uses parent `9dd63d992a5925dba7f9bfd1aec9dbfb1f54b837` plus the identical GPU regression test; production code is unchanged in the control. `source/control-test.patch.gz` records that test-only addition. All regular tracked source files were checked with the archived source manifests before execution.

Tests ran on shared GPUs 1 and 3 of host A6000PRO, whose actual GPUs are NVIDIA RTX PRO 6000 Blackwell Server Edition, SM120, 96 GB. No unrelated process was stopped. The runtime uses PyTorch 2.11.0+cu130, Transformer Engine 2.20.2 and FLA 0.5.1. Runtime launchers and the existing benchmark scripts are archived under `runtime/`. Source code in the PRs was not changed for these runs.

**Hardware scope:** [TE 2.20.2 lists SM100 and SM103 for NVFP4 training](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html#supported-devices). Its runtime `check_nvfp4_support()` returns true on this SM120 machine; no support check was overridden. These results establish the recorded behavior on SM120 and do not replace validation on the documented training devices. They do not establish convergence or performance.

## Results

- `native/`: 18 complete eight-layer GDN/attention or GDN2/attention decoder-training cases pass. They cover GDN fused/unfused pre-GDR, norm recomputation and fractions 0/0.5/1, comparing loss, gradients, weights, full optimizer state and pool ownership over three optimizer steps. RHT and stochastic rounding are disabled before Python starts for stable comparisons; the existing tolerance is retained for fused GDN atomic gate gradients.
- `graph/`: nine single-GPU NVFP4 decoder-training cases pass with real TE attention/whole-layer graph capture and replay, including norm recomputation, fusion, delayed offload, fractions 0/0.5/1 and fused weight-gradient accumulation. Each case compares offload disabled/enabled for six steps, including complete training states and pool ownership. RHT and stochastic rounding are disabled for these replay tests.
- `gdn2-two-rank/` and `gdn2-pp2/`: two NVFP4 GDN2 graph-training cases pass per rank under TP2+SP and under PP2, respectively. Each topology covers attention and whole-layer capture, including a norm-recompute/delayed-offload case. These tests use the same constrained recipe and compare the complete training states.
- `qwen-{eager,graph-off,graph-repeat,graph-on}/`: the complete 24-layer Qwen3.5-0.8B language model (vision and MTP excluded) trains from local pretrained weights and JSONL data in four fresh processes. Sequence 128, four microbatches, threshold 1024 elements and six optimizer steps are identical across arms. **RHT and stochastic rounding are enabled**, and the backward override is unset. Input hashes, losses, gradient norms, all gradient and weight fingerprints and full optimizer-state fingerprints match exactly across all four arms. The graph arms capture 24 layers. The offload arm selects 31,612,928 transfer bytes and has no pinned buffers in use after each step. Parameter transpose caching is disabled, as required by the current NVFP4 graph offload configuration. FLA recurrence and offloaded tensors remain BF16/FP32.
- `alignment-control/` and `alignment-fixed/`: on two GPUs, the old layout fails both FP32/BF16 gradient-buffer cases per rank in real TE NVFP4 GEMM with `CUBLAS_STATUS_NOT_SUPPORTED`. #8556 passes both cases on both ranks, including DDP reduction, finite/nonzero gradients, repeat equality and cross-rank equality. This is a GPU regression test with a small real TE model, not a full pretrained-model training run. Its recipe explicitly disables RHT and stochastic rounding.

The selected pytest cases have no skips. These are correctness tests on a shared host, not performance measurements; the benchmark's fingerprinting work is included in its raw step timings. No high-precision backward override is used. Run `python3 -S audit.py` to verify checksums and the recorded results.

## Commands

All commands ran through the existing private runtime and `torch.distributed.run`; rank stdout/stderr, pytest reports and launcher logs are preserved. `.log.gz`, `.patch.gz` and result JSON archives preserve the original bytes. The GPU-1 default launcher enables RHT and stochastic rounding; the constrained GPU-3 and GPU-1/3 launchers disable them. The latter was derived from the existing GPU-1/2 launcher by changing only the GPU device/UUID selection. `gpu13_default.py` changes only the two recipe environment flags relative to `gpu13_constrained.py`.

Common setup:

```bash
B=/home/lwx/.cache/megatron-gdn-20261026
R=$B/nvfp4-current-20261103
export GDN_SOURCE=$B/conflict-20261103-precision
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONFAULTHANDLER=1
unset NVTE_BACKWARD_OVERRIDE
```

Native training uses the prior, private precision cache roots and GPU 3:

```bash
export GDN_INDUCTOR_CACHE_DIR=$B/conflict-20261103/cache/precision/inductor
export GDN_TRITON_CACHE_DIR=$B/conflict-20261103/cache/precision/triton
timeout 900 /usr/bin/python3 "$B/precision_graph_container_gpu3_constrained.py" \
  /home/songbinbin/.venvs/megatron-gdn-offload/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node=1 --log-dir "$R/native/ranks" --redirects 3 \
  "$B/rank_env_precision_graph.py" "$B/rank_distributed_pytest.py" "$R/native/reports" \
  -vv -s -p no:cacheprovider tests/unit_tests/ssm/test_gdn_offload_training.py -k fp4
```

The graph suite uses the same source and GPU-3 constrained launcher, cache roots `$B/conflict-20261103/cache/graph/{inductor,triton}`, one rank, a 900-second timeout, output `$R/graph`, and pytest arguments:

```bash
-vv -s -p no:cacheprovider --confcutdir=tests/unit_tests/pipeline_parallel \
  tests/unit_tests/ssm/test_gdn_offload_cuda_graphs.py -k 'fp4 and tp1'
```

The GDN2 TP2+SP suite uses the same source, `$R/runtime/gpu13_constrained.py`, cache roots `$R/cache/two-rank/{inductor,triton}`, two ranks, a 900-second timeout and output `$R/gdn2-two-rank`:

```bash
-vv -s -p no:cacheprovider --confcutdir=tests/unit_tests/pipeline_parallel \
  tests/unit_tests/ssm/test_gdn_offload_cuda_graphs.py \
  -k 'gdn2 and fp4 and not tp1 and not 0.0 and not 0.5 and False-tp'
```

The PP2 launch changes cache roots to `$R/cache/pp2/{inductor,triton}`, output to `$R/gdn2-pp2`, and the selection to `gdn2 and fp4 and pp2 and not 0.0 and not 0.5 and False-pp2`.

The Qwen arms use GPU 1 and cache roots `$R/cache/qwen/{inductor,triton}`. Add `--no-graph` for ARM=eager, no flag for ARM=graph-off or graph-repeat, and `--offload` for ARM=graph-on:

```bash
export GDN_INDUCTOR_CACHE_DIR=$R/cache/qwen/inductor
export GDN_TRITON_CACHE_DIR=$R/cache/qwen/triton
timeout 900 /usr/bin/python3 "$B/precision_graph_container_gpu1_default.py" \
  /home/songbinbin/.venvs/megatron-gdn-offload/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node=1 --log-dir "$R/qwen-ARM/ranks" --redirects 3 \
  "$B/rank_env_precision_graph.py" "$B/benchmark_nvfp4_controls.py" \
  --model qwen --precision fp4 --weights "$B/models/Qwen3.5-0.8B" \
  --data "$B/data/ks-item-sft.jsonl" --seq-length 128 --threshold 1024 \
  --check-state --warmup 1 --iterations 2 --output "$R/qwen-ARM/result.json"
```

The alignment cases use the GPU-1/3 constrained launcher and cache roots `$R/cache/alignment/{inductor,triton}`. Run once with `GDN_SOURCE=$B/nvfp4-alignment-control-9dd`, ARM=control, then with `GDN_SOURCE=$B/nvfp4-alignment-fixed-66b`, ARM=fixed. The control is expected to fail:

```bash
timeout 240 /usr/bin/python3 "$R/runtime/gpu13_constrained.py" \
  /home/songbinbin/.venvs/megatron-gdn-offload/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node=2 --log-dir "$R/alignment-ARM/ranks" --redirects 3 \
  "$B/rank_env_precision_graph.py" "$B/rank_distributed_pytest.py" "$R/alignment-ARM/reports" \
  -vv -s -p no:cacheprovider \
  tests/unit_tests/distributed/test_distributed_data_parallel.py::TestDistributedDataParallel::test_nvfp4_fused_weight_gradient_alignment
```

## Complete Qwen TP2 baseline failure

`qwen-tp2-eager/` is an uninstrumented, offload-disabled, eager full-Qwen TP2+SP baseline at the same #8543 source. RHT and stochastic rounding are enabled, weight-gradient accumulation fusion is disabled, and no backward override is set. Both ranks fail during the first backward with TE/cuBLAS `CUBLAS_STATUS_NOT_SUPPORTED`; zero optimizer steps complete. This is not counted as a passing Qwen TP2 training path.

`qwen-tp2-shapes/` reruns that baseline with a diagnostic wrapper that delegates the original GEMM unchanged and logs its metadata only after a RuntimeError. On both ranks, TE's dgrad call receives `NVFP4TensorStorage` operands of shapes `[4112, 1024]` and `[128, 4112]`, layout `NN`, BF16 output, and no caller-provided output buffer. The reduction dimension is 4112. This is separate from #8556's preallocated weight-gradient buffer alignment.

`shape-repro/` uses only PyTorch and TE, with no Megatron import or offload hooks: BF16 `te.Linear(1024, width)` under the default NVFP4 recipe, a differentiable `[128,1024]` input, and an ordinary backward. Width 4112 reproduces the cuBLAS failure; width 4096 and 4128 forward/backward controls pass with finite input and weight gradients. These observations isolate a shape-related TE/cuBLAS limitation in this tested stack. They are consistent with the [cuBLAS 13.0 FP4 matrix-alignment requirements](https://docs.nvidia.com/cuda/archive/13.0.0/cublas/index.html#using-the-cublaslt-api), but do not establish behavior on SM100/SM103. No padding or precision fallback was added to the PRs.

Two preliminary metadata-logging attempts are retained under `qwen-tp2-trace/` and `qwen-tp2-trace-retry/`. The first failed to preload Torch's CUDA/NCCL dependencies before importing TE; the second tried to read a nonexistent public dtype property on TE storage. These diagnostic failures are not training results or evidence of a PR defect. The corrected wrapper is `runtime/trace_gemm.py`; the authoritative uninstrumented baseline and its successful metadata capture are listed above.

The uninstrumented TP2 baseline uses the same source, GPU-1/3 default launcher, a 900-second timeout, cache roots `$R/cache/qwen-tp2/{inductor,triton}`, and output `$R/qwen-tp2-eager`:

```bash
timeout 900 /usr/bin/python3 "$R/runtime/gpu13_default.py" \
  /home/songbinbin/.venvs/megatron-gdn-offload/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node=2 --log-dir "$R/qwen-tp2-eager/ranks" --redirects 3 \
  "$B/rank_env_precision_graph.py" "$B/benchmark_qwen_graph_tp2.py" \
  --model qwen --precision fp4 --weights "$B/models/Qwen3.5-0.8B" \
  --data "$B/data/ks-item-sft.jsonl" --seq-length 128 --threshold 1024 \
  --check-state --warmup 1 --iterations 2 --no-graph --output "$R/qwen-tp2-eager/result.json"
```

The metadata launch uses the same command with a 600-second timeout, output/log directory `$R/qwen-tp2-shapes`, and inserts `"$R/runtime/trace_gemm.py"` immediately before the benchmark path. The helper re-raises the original GEMM exception. Both launches retain the original per-rank cuBLAS errors; they are recorded separately from passing test reports.

The standalone shape reproduction uses `GDN_SOURCE=$B/conflict-20261103-precision`, cache roots `$R/cache/shape/{inductor,triton}`, `NVFP4_SHAPE_OUTPUT=$R/shape-repro/result.json`, the GPU-3 default launcher, a 180-second timeout and no distributed/model wrapper:

```bash
unset NVTE_BACKWARD_OVERRIDE
timeout 180 /usr/bin/python3 "$B/precision_graph_container_gpu3_default.py" \
  /home/songbinbin/.venvs/megatron-gdn-offload/bin/python "$R/runtime/repro_te_shape.py"
```
