# GDN PR scope reduction and GPU acceptance

Main baseline: `412eacc96331b9ea622598d9022eb6cb8b66ff85`.

The scope reduction removes experimental profiler/Bridge/tokenizer/checkpoint tools
and their tool-only tests from feature PRs. It retains native GPU regression,
backward-anchor, storage-alias, delayed-copy and pre-reset pool-ownership coverage.
No FLA or pre-GDR numerical kernel is changed. #8244 stays closed.

## Source identities and the #7399 prerequisite

| Increment | PR source | Executed GPU composition |
| --- | --- | --- |
| shared | `7dd45a0b96f16743f7988aa822205e1c08edfad8` | `7dd45a0b96f16743f7988aa822205e1c08edfad8` |
| feature | `edcb3c98745144fe2aa2868ff1bf41265dde31f6` | `c69d888227ea39b819ec182491449e8f1e49d0e5` |
| gdn2 | `3576a2fb1268e7a8b56e32be38b2b4e01359b725` | `0229d274b3e5ea0e8abfa5f836b7eb3a6393b2d2` |
| graph | `f858ea43cecc3a6a964f452b4cc094d87d16baf9` | `09abdaea1c30d67693998908c70cf54ae7b37a5d` |
| precision | `504d24c92b7471b6a7216e1169b6e291b0ce73dd` | `bb898e370d2bd4900d98298eed5782893d3111cd` |
| mrope | `549f1f6729773fbe1e954a47cf3ea0da25c97f1a` | `549f1f6729773fbe1e954a47cf3ea0da25c97f1a` |
| alignment | `d86af17c59cafd24f1f327eb25290fe43f8aa1b8` | `d86af17c59cafd24f1f327eb25290fe43f8aa1b8` |

For feature, GDN2, graph and low precision, the executed composition adds only
`megatron/core/pipeline_parallel/fine_grained_activation_offload.py` from upstream
#7399 at `aba09a62863cc8ba592da55a4c2fcb28ca0b5b86`. The exact manager-only patch is
`pr-7399-prerequisite.patch.gz`; source diffs and regular-file checksums are under
`source/`. The composed commits are available on the fork's
`validation/gdn-scope-20261123/{feature,gdn2,graph,precision}` branches.
**These are prerequisite-composed results, not claims that the GDN heads pass
against unmodified main without #7399.** The feature PRs do not duplicate its code.

The standalone GPT regression in #8243 passes on main without that prerequisite.
It demonstrates coverage, not a new scheduler defect: the old fraction-zero crash
no longer reproduces after #7575. A deliberately delayed D2H boundary test in the
GDN suite exposes the known inline-reload ordering dependency. That stress result
is not presented as a newly discovered normal-training bug in this work.

## Recorded acceptance results

| Result directory | GPUs / cases per rank | Result |
| --- | --- | --- |
| `shared-acceptance` | A6000 ×1 / 3 | All pass: native GPT, fractions 0/0.5/1, real copy observation and loss/gradient comparison |
| `feature-acceptance-fixtures` | A6000 ×1 / 47 | All pass: configuration, lifecycle and full decoder/DDP/Adam comparisons |
| `feature-acceptance-two-rank` | A6000 ×2 / 4 | All pass: TP2+SP, CP2, PP2 and PP2+VPP2 native training |
| `gdn2-acceptance-fixtures` | A6000 ×1 / 12 | All pass: gate-storage ownership, fractions, nonuniform packed/norm and decoder training |
| `graph-acceptance-single` | A6000 ×1 / 5 | All pass: mixed attention, fused/norm/delay, both whole-layer variants and fraction zero |
| `graph-acceptance-two-rank` | A6000 ×2 / 2 | All pass: one positive-transfer TP2+SP case and one PP2 case |
| `precision-acceptance-complete` | RTX PRO 6000 ×1 / 21 | All pass: 8 config, 6 cache-kwargs, 4 native-training and 3 graph-training cases |
| `mrope-acceptance` | A6000 ×1 / 1 | Pass: one capture, positions A → B → A, eager loss and named main-gradient comparison |
| `mrope-main-control-fixtures` | A6000 ×1 / 1 | Expected failure: identical regression on unmodified-main production code loses position information |
| `alignment-release` | RTX PRO 6000 ×2 / 35 | All pass: layout tests plus real TE/DDP NVFP4 wgrad for FP32/BF16 gradient buffers |
| `alignment-control-release` | RTX PRO 6000 ×2 / 2 | Expected failures on both ranks: identical test with old main layout, `CUBLAS_STATUS_NOT_SUPPORTED` |

The selected acceptance cases have no skips. Reports retain each node ID and
outcome, distinguishing collection from execution. All acceptance launches use
project fixtures. The numerical
operations, communication, hooks, copies, loss and optimizer steps are real.
Observers call the original production methods. Delay injection is confined to
the explicitly named delayed-D2H stress test; no scheduling queue is edited.

`source.json` records the source commit/status where available; independently
committed fixes also have archived commit IDs and patches. Negative controls add
only the exact regression test to the main production tree; their test-only
patches are archived. Main-gradient comparisons use parameter names. mRoPE uses
`rtol=1e-5, atol=1e-6`; native fused GDN uses the existing explicit tolerance for
atomic gate gradients. Other recorded training state comparisons are exact.

The low-precision graph cases compare six optimizer steps per arm; NVFP4 compares
eager, graph without offload and graph with offload. They require replayed layers,
positive transfer selection and released pinned buffers before reset. Native LP
training uses unfused wgrad; it does not substitute for #8556. The wgrad regression
asserts that the odd-sized marker and TE weight share a bucket. Repeated cleared
backward passes are not a multi-microbatch accumulation qualification.

## Environment and reproduction

Local: NVIDIA RTX A6000 SM86, using the existing project Docker runtime.
Remote: A6000PRO's NVIDIA RTX PRO 6000 Blackwell Server Edition SM120. GPU device
UUIDs and available memory are retained in per-run preflight metadata. Shared
GPUs with sufficient free memory were authorized; no unrelated process was killed.
Runtime: PyTorch 2.11.0+cu130, TE 2.20.2, FLA 0.5.1. The fused GDN cases use the
existing causal-conv1d 1.6.1 dependency. Exact launcher arguments are preserved in
`results/*/command.json`, and rank stdout/stderr are preserved byte-for-byte.

The final NVFP4 launchers disable RHT and stochastic rounding before Python starts
for exact comparisons. They unset `NVTE_BACKWARD_OVERRIDE`. No support check is
bypassed. Runtime support on SM120 does not replace SM100/SM103 qualification;
[TE's training-device documentation](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html#supported-devices)
lists those devices. No performance or convergence qualification is claimed.
The earlier width-4112 NVFP4 dgrad failure is separate, remains at its recorded
source revision in `nvfp4-current-20261103`, and is not fixed by this work.

To reproduce, check out the PR source or recorded composition, provide the normal
TE/FLA dependencies and run the pytest node IDs from the command JSON through
`python -m torch.distributed.run`. The archived launchers show the exact local
container and remote private-runtime paths used here; adapt those paths to the
host. Archived source checksums allow a third party to check the copied source.

Black, isort, Ruff and project Pylint checks pass on the changed files; the
full autoformatter command and kernel determinism-coverage output are retained
under `results/lint`. GPU CI in NVIDIA's infrastructure has not run.

## Intermediate attempts and exclusions

Earlier directories in `results/` are retained for diagnosis, not added to the
acceptance counts above. They include:

- an initial local runtime C++ ABI problem and GPU-selection mistakes: numeric
  container indices mapped to an occupied card; selecting a host UUID absent
  from that container exposed no CUDA device. Later launches use available,
  exposed UUIDs. Resource failures are not evidence of a PR bug;
- a mixed-scope assertion exposing the shared attribute name between ordinary
  attention and GDN, followed by the distinct GDN flag;
- a missed capture-side read during that rename, causing unjoined graph work
  and subsequent RNG errors. The final code fixes that read; it adds no stream
  fallback or scheduler workaround;
- a kwargs test missing RNG seeding and an invalid BF16 repeat-loss assumption
  applied to evolving low-precision scaling metadata. These were test errors,
  corrected without changing numerical production code;
- an initial NVFP4 acceptance sync that had all executed core/test files but
  omitted some unrelated trailing tools during concurrent checkout. The
  `precision-acceptance-complete` rerun uses the complete, checksum-verified source.

`reproduction/81b5e39259bbebbcc951d1e5e269ed3ce190bde5/` retains the removed
experimental tools and their tests at that exact historical source. Original
memory/runtime reports keep their recorded SHAs and do not qualify the revised
heads. GDN2 eager, TE graphs, mRoPE capture, low-precision enablement and NVFP4
wgrad address alignment are separate increments with separate evidence.

Run `python3 -S audit.py` to check archive integrity, passing acceptance reports,
expected negative controls, source identities and the fixed-version tool archive.
