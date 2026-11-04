# DCP statistics: current-source benchmark and DP2 acceptance

Validation date: November 3, 2026, America/Los_Angeles (training logs use UTC).
Tested PR source: `79bf7b6bcaa95538d2ad7068dc46b7a931fd2c7d`, based on dev
`7b66fcc088e6e60b87c350e0e185721e0373d0ab`, PR #8431. No production code or
compatibility patch changed during these runs. `source.json` and
`tested-file-hashes.json` identify the tested source.

This supplements the same-source TP2/PP2/VPP2 training records in
[the earlier rebase validation](https://github.com/WEIXILIN0307/Megatron-LM/tree/af86402bcd69ca0926bd81af7ffc1978e5f7ee72/validation/dcp-stats-rebase-20261103).
It closes the missing current-source benchmark and actual DP2 logging-window
coverage. These are small dense GPT BF16 training runs with synthetic,
length-controlled tokens; they do not establish production convergence or scaling.

## Current-source benchmark

Real `pretrain_gpt.py` forward/backward and Adam updates, two RTX A6000 GPUs,
TP1/PP1/DP1/CP2. Four layers, hidden 128, FFN 512, four heads, vocabulary 256,
sequence limit 1024, zero dropout. All arms restore the same checkpoint,
optimizer, scheduler and RNG state, use seed 1234 and train 2,560 effective tokens
per update. The fixed-CP arm uses `dp_balanced`; the DCP arms use
`default_dynamic_cp`, differing only in `--log-dcp-stats` and output paths.

Three rotated repetitions per arm:
`off/on/fixed`, `on/fixed/off`, `fixed/off/on`. Each run performs 100 updates,
iterations 2–101, discards the first ten, and retains 90 steady updates. Existing
`interval-time` timings include normal stdout and TensorBoard logging at interval
1. W&B is off. The primary runs have no tensor hashing or CPU profiling.
Process CPU affinity is fixed and recorded in each `command.sh`. This is a shared
host without locked GPU clocks; `current-gpu.csv` retains activity, clocks,
temperatures and power for both measurement GPUs.

| Arm | Mean step ± between-run sample std (ms) | Mean effective tokens/s |
|---|---:|---:|
| Fixed CP2 | 142.42 ± 0.40 | 17,975 |
| DCP, metrics off | 117.61 ± 5.39 | 21,796 |
| DCP, metrics on | 116.66 ± 2.25 | 21,949 |

On/off mean step-time difference: **−0.81%**. The paired differences are
+1.20%, −3.79%, +0.39%. These observations do not resolve small overhead, prove
zero overhead, or show that enabling metrics speeds up training. DCP versus fixed
CP is a comparison of the existing schedulers on this workload; the metrics PR
does not change either scheduling decision.

`current-performance.json` retains per-run means, medians, per-step standard
deviations, trimmed means, tokens/s and paired differences. Tokens/s is calculated
per run from total effective tokens divided by measured time, then averaged across
runs. `verify_performance.py` confirms checkpoint restoration, exactly 100 updates,
identical token/microbatch counts and exactly one stdout/TensorBoard event per
metric per enabled update. Disabled/fixed runs have no DCP events.

## Separate CPU diagnostic

`cpu-profile-off` and `cpu-profile-on` each restore the checkpoint and perform
40 updates, discarding ten, leaving 30 diagnostic samples per rank. Python
`perf_counter_ns` wrappers measure the helper, accumulator and existing logger;
they add no operator timing or GPU synchronization. These runs are excluded from
the benchmark aggregates.

Enabled statistics computation averages 40.14 / 41.43 µs per step on the two
ranks; accumulator update averages 12.62 / 13.30 µs. Disabled diagnostic runs make
no statistics-helper or accumulator-update calls. Emitting-rank `training_log`
averages 1.619 ms off / 1.535 ms on; it includes existing logging/synchronization
and host variation, so that difference is not a speedup claim. All per-call
samples and `cpu-profile-summary.json` are retained.

## Real DP2×CP2 and logging windows

Four A6000 GPUs, TP1/PP1/DP2/CP2: **four DP×CP ranks**. The varying lengths in
`dp2-lengths.csv` exercise different total work, CP distributions and microbatch
counts. An initial one-update checkpoint is restored for both off/on arms.
Each arm performs nine updates, iterations 2–10, with log interval 3 and
interleaved evaluation. `train_audit.py` records final schedules, lengths,
returned summaries, loss/gradient/parameter hashes and accumulator snapshots
around logging and evaluation. Auditing is excluded from performance runs.

All nine updates match bitwise in final schedules, scheduled microbatch counts,
losses, gradient hashes and updated parameter hashes. `verify_dp2.py` independently
recomputes rank work, per-microbatch conservation, global summaries, and windows;
it does not import the production statistics helper for expected results. Every
legal microbatch conserves sum of unique logical-sample padded lengths squared.
All four ranks hold the same global summary; each successful update contributes
eight logical samples globally, rather than eight per rank.

| Logged iteration | Summed numerator | Summed denominator | CP1 / CP2 logical samples | Balance |
|---|---:|---:|---:|---:|
| 3 | 2,621,440 | 2,752,512 | 13 / 3 | 0.9523809524 |
| 6 | 6,881,280 | 6,881,280 | 20 / 4 | 1.0000000000 |
| 9 | 5,210,112 | 6,520,832 | 18 / 6 | 0.7989949749 |

TensorBoard emits only iterations 3, 6 and 9, once per metric. The existing logger
also prints the first resumed iteration (2) to stdout without resetting the
window. Iteration 10 remains as the final unlogged partial window. Completed
windows reset, and evaluation leaves the accumulator unchanged. The JSON report
and verifier explicitly check these boundaries. CP counts are summed before
fractions are calculated. The existing unit tests cover unequal sample-count
windows as well.

## Regressions and loop without updates

The four-rank focused regression passed **134 tests per rank**, 31 deselected.
The recorded filter excludes existing other-topology cases and four unrelated
optional-import model-builder cases. The full unit suite was not run. Coverage
includes the existing formula, VPP, TP propagation, logger ownership, weighted
window/reset, rerun, skipped/aborted/exception/evaluation and profiling tests.
See `unit-command.txt` and `pytest.log`.

The actual `run_train_loop_without_updates` path was rerun for three iterations
on two GPUs with metrics enabled and `--no-load-optim`. Its audit fails if
`train_step` or schedule statistics run. The data hook ran each iteration,
parameters were unchanged, all submitted windows were empty, and stdout/TensorBoard
had no DCP metrics. `verify_no_updates.py` and `no-updates-check.json` check the
recorded results.

W&B service integration, internally padded Ampere THD, MoE, CUDA Graph and
production-model convergence/scaling remain unverified. A repository functional
recipe has not been registered; the retained GPU replay scripts are manual
acceptance evidence. Upstream CI and maintainer review are separate merge gates.

## Reproduction and provenance

Image `megatron-dcp-stats:validation-20261022`, SHA
`070e339faa1fb4fec1c630f7bcc77c97ebd7eb138bc81802feac567aa011bcb8`.
`environment.json`, `container-command.txt`, GPU/CPU topology files and each
run's `command.sh` retain versions and exact commands. Environment: Python 3.12.4,
Torch 2.11.0+cu130, TE 2.20.2, FA2 2.8.3, cuDNN 9.19.0, NCCL 2.28.9,
driver 580.178.04. Adapt paths, CPU affinity and GPU indices to the host.

1. Check out the tested source. Copy the replay scripts and both lengths CSVs
   into a fresh artifact directory. Create the container using the recorded
   template with that directory mounted. Set `DCP_REPO_ROOT` to the checkout.
2. Create `initial-aligned` with `launch_train.py initial-aligned --iters 1
   --save --container <container>` using two GPUs. The measurements here reuse
   the identical checkpoint from the earlier same-source validation, copied
   into the mounted artifact directory. Binaries are omitted; checkpoint file
   hashes are retained in `checkpoint-hashes.json`.
3. Set `DCP_CONTAINER`, `DCP_GPU_INDICES=4,5`, `DCP_PERF_PREFIX=current`,
   `DCP_PERF_ITERS=101` and the recorded `DCP_MEASURE_CPU_BINDINGS`. Run
   `python -S run_performance.py`, then `python -S analyze_performance.py current`
   and `verify_performance.py` in the container.
4. Run `run_diagnostics.py`, then `analyze_cpu_profile.py`. Diagnostic wrappers
   are enabled only for these separate runs.
5. Run `run_dp2.py` on the host and `verify_dp2.py` in the container. The runner creates a
   fresh topology-specific checkpoint, then the off/on pair. `DCP_GPU_INDICES`
   is `1,4,5,6` for the recorded four-rank runs.
6. Run `launch_train.py no-updates --no-updates --on --iters 3 --container
   <container>` with two GPUs, then `verify_no_updates.py`. Run the focused
   pytest command from `unit-command.txt` with four GPUs.
7. Run `sha256sum -c SHA256SUMS` from the published evidence directory.

`setup-attempts/checkpoint-not-mounted` retains the initial **invalid** timing
attempts. A checkpoint symlink pointed outside the mounted container directory,
and the loader warned before starting from iteration 0. Those nine runs are
excluded from every reported aggregate. The corrected attempts copy the
checkpoint into the mounted directory, set `--exit-on-missing-checkpoint`, and
require successful restoration in the analyzer. Keeping the rejected attempts
makes this setup correction auditable.

The helpers adapt the existing acceptance scripts; no training or benchmark
framework is added to the metrics PR. `SHA256SUMS` covers every retained evidence
file except the manifest itself.
