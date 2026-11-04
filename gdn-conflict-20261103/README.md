# Activation-offload validation after the upstream scheduling change

All candidates include NVIDIA `main` at `d00e888f270e2270da872a9679d490f9411e4ece`, including #7575, and the latest #7399 prerequisite at `6bd691578a7f8b24640056fb6630ca9557be40e5`.

| PR | Tested source commit |
| --- | --- |
| #8243 | `351575875328cc17ae828f27949f038ad1246c9b` |
| #7908 | `89a33edd415a84bcff3628349f061baf3e346e83` |
| #8308 | `9e4e34eda8fe07b8a5ebacfca4a3aeb07f6cc796` |
| #8543 | `81b5e39259bbebbcc951d1e5e269ed3ce190bde5` |

The source trees were copied from isolated Git worktrees. Every regular tracked file in the remote trees was verified with `sha256sum -c source/NAME.sha256`; the check output is archived beside the manifests. `source/*.json` identifies each revision. `pr-*-diff.patch.gz` records the incremental changes; the #8243 diff against main includes #7399's production changes.

The original #8243 fraction-zero failure no longer reproduces on this main. After #7575, resident groups retain reload slots and warmup respects the fraction. The candidates restore upstream `finish_all_groups()`; #8243 now contributes its native multi-microbatch training regression, rather than an additional production fix. #7399's inline reload ordering remains a prerequisite and is composed with the new resident-group schedule.

## Environment and results

Tests ran on shared RTX PRO 6000 GPUs (SM120), using the private mount-namespace runtime in `runtime/`. `environment.json` records Python 3.12.4, PyTorch 2.11.0+cu130, TE 2.20.2, FLA 0.5.1 and Triton 3.6.0. No unrelated process was stopped. These are correctness checks; concurrent GPU and host workloads prevent performance conclusions.

- `baseline-qknorm/`: the original native four-layer GPT regression passes at fractions 0, 0.5 and 1 on main without candidate production changes. Two microbatches and three SGD steps compare losses and all gradients against offload disabled. The only source change is the archived `source/baseline-test.patch.gz`. Warmup transfer expectations follow #7575; Q/K normalization follows the upstream offload tests and avoids their newly rejected packed Q/K/V layout.
- `shared-final/`: 19 selected GPU tests pass, including the native training regression, upstream partial-offload reload cadence and margin-group tests, warmup accounting, #7399's ordered inline reload and checkpoint storage exclusion.
- `scope-final/` and `training-final/`: 71 scope/configuration tests and 12 complete GDN/attention decoder-training cases pass in separate processes. The training cases cover fractions 0/0.5/1, norm recomputation and native fused/unfused pre-GDR. There are no skips. `combined-recheck/` reruns the scope-to-training transition in one process (62 passes). `combined-full/` reruns the original combined selection in one process (73 passes). Its final report is checked by the audit.
- `feature-two-rank/`: four complete decoder-training cases pass per rank, covering TP2+SP and PP2, offload fractions 0 and 1, three BF16 Adam steps, losses, accumulated gradients, weights, full optimizer states and pool ownership. There are no topology skips in these selected cases.
- `qwen-feature-{off,zero,on}/`: the complete 24-layer Qwen3.5-0.8B language model, excluding vision and MTP, trains from local pretrained weights and JSONL data in three fresh processes. BF16, sequence 128, four microbatches, norm recomputation, threshold 1024 elements and three optimizer steps are identical across arms. All recorded loss, gradient, weight and complete optimizer-state fingerprints match exactly. Disabled and fraction-zero arms select no transfers; fraction one selects 132,030,464 bytes. The training tool checks pinned-pool ownership before resets.
- `graph/`: seven single-GPU GDN/GDN2 decoder-training cases pass with real TE graph capture/replay. They cover attention/whole-layer capture, norm recomputation, fusion, delayed offload and fraction zero, comparing offload disabled/enabled and complete optimizer states.
- `precision/`: two GPU graph-training cases pass: GDN FP8 delayed scaling and GDN2 FP8 tensorwise scaling. `precision-config/` records 19 passing configuration checks; those checks do not themselves train a model.

Black, isort, Ruff, Pylint and the determinism coverage checker pass for the cumulative precision branch; their logs are archived here. No numerical kernel or autotuning policy was changed.

`baseline/` preserves the preliminary no-Q/K-normalization launch: fraction zero passed, while positive fractions hit upstream's explicit packed-Q/K/V rejection. This is not a candidate regression. Its test patch is `source/baseline-initial-test.patch.gz`. `feature/` preserves the first broad run interrupted by its 600-second timeout during cold compilation; it has no final passing report and is not counted. Its retry in `feature-final/` terminated with SIGSEGV and has no passing report. Separate scope/training runs and the one-process transition recheck pass; the cause of that earlier SIGSEGV has not been established. The original combined selection is rerun under `combined-full/` with verbose reporting and Python fatal-signal traceback enabled. Diagnostic stack dumps in graph/precision logs show FLA compilation and autotuning and do not indicate a test failure when the final report passes. The optional `/opt/data` download is unavailable in the private runtime; the selected tests do not depend on it.

These checks do not qualify evaluation-first initialization, variable shapes after warmup, a current four-rank matrix, full-Qwen graph/FP8 training on these commits, or NVFP4 on supported training GPUs. Historical stability, checkpoint and performance results remain tied to their original commits. Upstream GPU CI has not run.

Run `python3 -S audit.py` to verify SHA256SUMS, per-rank pytest reports and the Qwen comparisons.

## Commands

The original per-rank stdout/stderr and launcher outputs are preserved as `.log.gz`; result JSON files are compressed without changing their contents. The namespace launchers, rank environment/report wrappers and diagnostic wrapper are copied verbatim under `runtime/`.

Common setup on the test machine:

```bash
B=/home/lwx/.cache/megatron-gdn-20261026
R=$B/conflict-20261103
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
```

The final shared test uses `GDN_SOURCE=$B/conflict-20261103-shared`, cache roots `$R/cache/shared-final/{inductor,triton}`, the GPU-1 default launcher, one rank, output `$R/shared-final`, and this selection:

```bash
export GDN_SOURCE=$B/conflict-20261103-shared
export GDN_INDUCTOR_CACHE_DIR=$R/cache/shared-final/inductor
export GDN_TRITON_CACHE_DIR=$R/cache/shared-final/triton
timeout 180 /usr/bin/python3 "$B/precision_graph_container_gpu1_default.py" \
  /home/songbinbin/.venvs/megatron-gdn-offload/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node=1 --log-dir "$R/shared-final/ranks" --redirects 3 \
  "$B/rank_env_precision_graph.py" "$B/rank_distributed_pytest.py" "$R/shared-final/reports" \
  -q -s -p no:cacheprovider tests/unit_tests/pipeline_parallel/test_fine_grained_activation_offloading.py \
  -k 'resident_groups_preserve_forward_chunks or partial_offload_backward_does_not_accumulate or margin_groups_allow_prefetch or warmup_fraction_counts or inline_reload_is_ordered or checkpoint_without_output_storage'
```

The baseline uses the same launcher with `GDN_SOURCE=$B/conflict-20261103-baseline`, cache roots `$R/cache/{inductor,triton}`, a 480-second timeout, output `$R/baseline-qknorm`, and only `tests/unit_tests/pipeline_parallel/test_fine_grained_activation_offloading.py::test_offload_resident_groups_preserve_forward_chunks`.

The original combined retry (failed with SIGSEGV; retained for diagnosis):

```bash
export GDN_SOURCE=$B/conflict-20261103-feature
export GDN_INDUCTOR_CACHE_DIR=$R/cache/inductor
export GDN_TRITON_CACHE_DIR=$R/cache/triton
timeout 1200 /usr/bin/python3 "$B/precision_graph_container_gpu1_default.py" \
  /home/songbinbin/.venvs/megatron-gdn-offload/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node=1 --log-dir "$R/feature-final/ranks" --redirects 3 \
  "$B/rank_env_precision_graph.py" "$B/conflict-20261030/watch_run.py" \
  "$B/rank_distributed_pytest.py" "$R/feature-final/reports" \
  -q -s -p no:cacheprovider tests/unit_tests/ssm/test_gated_delta_net_offloading.py \
  tests/unit_tests/ssm/test_gdn_offload_training.py -k 'not tp and not cp and not pp'
```

The two-rank feature test uses the GPU-1/2 launcher, separate rank-owned compiler caches and this narrower topology selection:

```bash
export GDN_INDUCTOR_CACHE_DIR=$R/cache/feature-two-rank/inductor
export GDN_TRITON_CACHE_DIR=$R/cache/feature-two-rank/triton
timeout 720 /usr/bin/python3 "$B/precision_graph_container_gpu12_apex.py" \
  /home/songbinbin/.venvs/megatron-gdn-offload/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node=2 --log-dir "$R/feature-two-rank/ranks" --redirects 3 \
  "$B/rank_env_precision_graph.py" "$B/conflict-20261030/watch_run.py" \
  "$B/rank_distributed_pytest.py" "$R/feature-two-rank/reports" \
  -q -s -p no:cacheprovider tests/unit_tests/ssm/test_gdn_offload_training.py \
  -k '(tp_sp or pp) and not (tp_sp_cp or tp_sp_pp or pp_vpp) and unfused and False and (0.0 or 1.0)'
```

The Qwen arms use `GDN_SOURCE=$B/conflict-20261103-feature` and cache roots `$R/cache/qwen-feature/{inductor,triton}`. For ARM=off omit the fraction flag, for ARM=zero use `--fraction 0`, and for ARM=on use `--fraction 1`:

```bash
export GDN_SOURCE=$B/conflict-20261103-feature
export GDN_INDUCTOR_CACHE_DIR=$R/cache/qwen-feature/inductor
export GDN_TRITON_CACHE_DIR=$R/cache/qwen-feature/triton
timeout 600 /usr/bin/python3 "$B/precision_graph_container_gpu1_default.py" \
  /home/songbinbin/.venvs/megatron-gdn-offload/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node=1 --log-dir "$R/qwen-feature-ARM/ranks" --redirects 3 \
  "$B/rank_env_precision_graph.py" "$GDN_SOURCE/tools/ssm/qwen_gdn_offload_training.py" \
  --weights "$B/models/Qwen3.5-0.8B" --data "$B/data/ks-item-sft.jsonl" \
  --seq-length 128 --microbatches 4 --min-offloaded-tensor-size 1024 --check-state \
  --recompute-norm --warmup 1 --iterations 2 --output "$R/qwen-feature-ARM/result.json"
```

The graph suite uses `GDN_SOURCE=$B/conflict-20261103-graph`, cache roots `$R/cache/graph/{inductor,triton}`, the GPU-3 default launcher, one rank, a 720-second timeout, output `$R/graph`, the same diagnostic/rank wrappers as the feature test, and pytest arguments:

```bash
-q -s -p no:cacheprovider --confcutdir=tests/unit_tests/pipeline_parallel \
  tests/unit_tests/ssm/test_gdn_offload_cuda_graphs.py -k tp1
```

The FP8 suite uses `GDN_SOURCE=$B/conflict-20261103-precision`, cache roots `$R/cache/precision/{inductor,triton}`, the GPU-2 launcher, one rank, a 720-second timeout, output `$R/precision`, the same wrappers, and:

```bash
-q -s -p no:cacheprovider --confcutdir=tests/unit_tests/pipeline_parallel \
  'tests/unit_tests/ssm/test_gdn_offload_cuda_graphs.py::test_gdn_offload_with_te_graphs[gdn-False-False-False-False-1.0-fp8-delayed-False-tp1]' \
  'tests/unit_tests/ssm/test_gdn_offload_cuda_graphs.py::test_gdn_offload_with_te_graphs[gdn2-False-False-False-False-1.0-fp8-tensorwise-False-tp1]'
```

The configuration check uses the same precision source/caches and GPU-2 launcher, no diagnostic wrapper, one rank, a 120-second timeout, output `$R/precision-config`, and:

```bash
-q -s -p no:cacheprovider tests/unit_tests/ssm/test_gated_delta_net_offloading.py \
  -k 'test_reject_unsupported_gdn_offload or test_gdn_offload_config or test_gdn_offload_te_graph_config'
```

The successful independent scope launch uses the GPU-1 default launcher and rank wrappers, the same feature source and `$R/cache/{inductor,triton}`, `PYTHONFAULTHANDLER=1`, a 480-second timeout, output `$R/scope-final`, and the whole `tests/unit_tests/ssm/test_gated_delta_net_offloading.py` file. The independent training launch uses cache roots `$R/cache/training-final/{inductor,triton}`, a 720-second timeout, output `$R/training-final`, and `-vv -s -p no:cacheprovider tests/unit_tests/ssm/test_gdn_offload_training.py -k dp`. Neither uses the periodic diagnostic wrapper.

The transition recheck uses the same launcher/source/caches as the scope launch, output `$R/combined-recheck`, and `-vv -s -p no:cacheprovider tests/unit_tests/ssm/test_gated_delta_net_offloading.py 'tests/unit_tests/ssm/test_gdn_offload_training.py::test_gdn_offload_training[dp-unfused-False-0.0]' -k 'not tp and not cp and not pp'`. The full combined recheck uses output `$R/combined-full` and replaces the explicit training node with the entire training file. Both use `PYTHONFAULTHANDLER=1`, a 480-second timeout and the ordinary rank wrappers without the periodic diagnostic wrapper.
