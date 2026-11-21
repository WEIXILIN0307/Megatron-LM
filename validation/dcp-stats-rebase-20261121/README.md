# DCP statistics rebase acceptance — November 21, 2026

PR #8431 runtime tested at `e3311dba708a30e8d8418e491d4e15003750bbd6`,
rebased onto **upstream/dev** `6559f64c27f2c547a3b48e18bd6f7326d980df94`.
The previous PR head `915ffdb5a4d89a3b3f9491ee4d93038cf0985ba2` is retained locally
at `backup/dcp-stats-before-dev-rebase-20261121`. No branch switch to main or separate
attention compatibility patch was used. This evidence records correctness,
not a new throughput or observation-cost benchmark.

## Scope and compatibility

- Keep #9585's planner early exit and hybrid metadata preparation unchanged.
  `scope-check.json` checks all existing definitions in `data_schedule_utils.py`,
  the `DefaultDynamicCPScheduler` class including VPP alignment, and the complete
  `pretrain_hybrid.py` against upstream. The source diff only adds observation,
  logging and the focused compatibility regression.
- Compute DCP estimates from the final schedule returned by
  `DpBalancedScheduler.get_groups_and_subsamples`, after VPP alignment, using
  scheduling **padded lengths**. The existing FLOPs helper remains unchanged
  and uses **real token lengths**.
- Enabled summaries still share one FP64 TP broadcast. Only received legacy
  metadata on non-TP0 ranks is converted to its original FP32 precision. TP0
  retains its original Python values; new DCP fields retain FP64 precision.
  Disabled metadata follows the existing broadcast path.
- The actual rebase conflict was in `training_log`'s parameters. Both upstream's
  `embedding_min_norm_metrics` and this PR's `dcp_stats` are retained. VPP
  assembly was not changed, so no additional PP2/VPP2 real-training run was
  required or claimed for this rebase.

## CPU regression

`cpu-command.txt` runs the existing sequence-packing tests with
`CUDA_VISIBLE_DEVICES=`. **21 passed, 68 deselected**, covering manual formulas,
conservation, empty work, actual VPP alignment, TP metadata, log-window weighting
and reset, successful/rerun/skipped/aborted/exception outcomes, and evaluation
exclusion. This is a focused selection, not a full unit-suite run.

The existing `test_dcp_schedule_stats_tp_metadata` now runs off/on within one
case and compares the scheduler's four-field return on each TP rank. Its real
lengths are 4096 and 1, giving square sum **16,777,217**. The broadcast stub
models the real behavior: source Python values are returned intact, while only
receivers deserialize the chosen tensor dtype. Both flags must return:

| TP rank | Real token sum | Legacy real squared sum |
|---|---:|---:|
| 0 | 4097 | 16,777,217 |
| 1 | 4097 | 16,777,216 |

Distinct padded lengths 1,048,579 and 513 verify that the new DCP numerator,
denominator and rank-work matrix use scheduling lengths and retain precision.
Disabled collection must never invoke the production statistics helper.

The revised test failed before the compatibility fix: enabled TP1 incorrectly
returned 16,777,217. `tp-metadata-before-fix.log` retains that regression, while
`cpu-tests.log` records the successful fixed run. The test also rejects rounding
TP0 or degrading the new DCP fields to FP32.

## Four A6000 real BF16 training

Topology: **TP2×CP2, PP1, DP1**, world size 4, DP×CP domain 2. GPUs 0/2/5/6
were used on a shared host; no other workloads were stopped. Four-layer dense
GPT, hidden 128, FFN 512, four attention heads, vocabulary 256, sequence limit
1024, zero dropout. The checkpoint was produced by one actual Adam update.
Both off/on arms restore that same topology-specific checkpoint,
optimizer/scheduler and RNG state, use seed 1234 and the same `lengths.csv`, and
perform **five further Adam updates**, iterations 2–6.

`train_audit.py` observes actual `pretrain_gpt.py`. It records final schedules,
returned legacy FLOPs values, actual `set_seqlen_stats_in_iteration` inputs,
losses, gradients, updated parameter hashes, pending/logging windows and eval
boundaries. Audit hooks do not change scheduler decisions. The extra reference
planner call and tensor hashing make these correctness runs unsuitable for
performance claims.

`verify_training.py` independently reconstructs work and CP counts from the
raw final schedules, checks conservation, compares off/on per rank, and checks
stdout and TensorBoard events. `correctness.json` records the results:

- All five updates match **bitwise** in final schedules, scheduled microbatch
  counts, legacy FLOPs return values and actual training inputs, losses,
  gradient hashes and updated parameter hashes. Every step updates parameters;
  gradient hashes are nonempty.
- Every rank receives the same legacy real-token inputs `[2560, 1507328]` under
  both flags. Three scheduled microbatches execute per update.
- Production DCP statistics run once per enabled training step on each TP source
  only; disabled training and evaluation make no such calls. TP peers receive
  identical summaries, without rank-work matrices.
- Each step has CP1/CP2 logical counts **7/1**, balance **0.92**, fractions
  **0.875/0.125**. Log interval 2 yields exactly one stdout record and TensorBoard
  event per metric at **iterations 2, 4 and 6**. Accumulated sample counts are
  8/16/16 across those windows, rather than multiplied by TP/CP replicas.
  Evaluation leaves the accumulator unchanged; completed windows reset.

No live W&B service, CUDA Graph, internally padded Ampere THD, production-model
convergence or scaling was validated. The determinism flag rejected by DCP is
omitted, while seed 1234, `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` and
`CUBLAS_WORKSPACE_CONFIG=:4096:8` are retained. Equality is observed for these
runs, not guaranteed for all workloads.

## Setup correction and checks

The first attempts reused the November 3 CSV, which encoded content lengths plus
one under the previous mock-dataset contract. Upstream now returns exactly the
CSV's content-token count, so that fixture produced internal THD padding
(e.g. 513 -> padded 520). TE's debug output confirmed FA2 rejects internal
padding on SM86. `setup-attempts` retains the failed logs, commands and original
CSV. The accepted fixture directly lists the intended aligned lengths
1024/512/256/128. **Only the acceptance data file changed; no training or TE
source was patched.** Both successful arms use this corrected file and a fresh
checkpoint.

Black/isort/Pylint/Ruff checks passed. Optional mypy reports **12 diagnostics
identical to the extracted upstream baseline**, without new messages. Logs and
`mypy-comparison.json` retain this comparison. Commit signatures and signoffs
are required for publication. Upstream CI and maintainer review remain separate
merge requirements.

## Replay

1. Check out runtime `e3311dba708a30e8d8418e491d4e15003750bbd6`. Copy the
   launch/audit/verifier scripts and corrected `lengths.csv` to a fresh artifact
   directory. Adapt the recorded container mounts and GPU indices to your host.
2. Start the container using `container-command.txt`, with the artifact directory
   mounted. Set `DCP_REPO_ROOT` to the checkout, `DCP_CONTAINER` to the container
   name and `DCP_GPU_INDICES` to four GPUs (recorded: `0,2,5,6`). Run
   `python -S run_training.py` **on the host** to create a fresh checkpoint and
   run the off/on pair. Each run's `command.sh` retains its exact command.
3. In the container, from the checkout, run `python /artifact/path/verify_training.py`.
   It regenerates `correctness.json` and verifies runtime-file hashes against the
   checkout. Checkpoint binaries are omitted from Git; their hashes are retained.
4. Run `cpu-command.txt` for the focused CPU tests. `autoformat.log` records the
   repository formatter invocation; `baseline/` contains the upstream source
   used for the optional mypy comparison.
5. From the evidence directory run `sha256sum -c SHA256SUMS`.

Image `megatron-dcp-stats:validation-20261022` (SHA in `image.txt`), Python 3.12.4,
Torch 2.11.0+cu130, TE 2.20.2, FA2 2.8.3, cuDNN 9.19.0 and NCCL 2.28.9.
`environment.json`, `gpu-snapshot.csv`, `source.json`, `tested-file-hashes.json`
and `range-diff.txt` retain the environment and source identity. No historical
benchmark is presented as a measurement of this runtime.
