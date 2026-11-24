# GDN test reduction and current-head GPU checks

Only `tests/unit_tests/ssm/test_gated_delta_net_offloading.py` changes relative
to the previous four PR heads. Production code, native decoder training,
graph replay, low-precision tests and independent fix regressions are unchanged.

## Retained behavior

| Test area | Previous cases | Retained cases and independent coverage |
| --- | --- | --- |
| Delayed H2D / microbatch accumulation | 6 | 2: partial offload and full offload with norm recomputation; fraction zero stays in recurrence replay and native decoder tests |
| Packed sequences | 4 | 2: unfused and fused + norm; both now use nonuniform lengths `[0, 32, 128]` |
| Evaluation/no-grad and training resumption | 15 | 3: eval/no-grad, eval/grad with fusion + norm, train/no-grad; each checks manager bypass before first training, bypass after policy learning and correct resumed backward |
| GDN2 replay policy | 4 | 2: resident and partial packed + norm; full offload remains in the gate-storage and native decoder tests |
| TE graph positive config readback | 2 | Removed: real GDN/GDN2 capture/replay already validates acceptance and behavior |

Twenty-two redundant parameter combinations are removed. Frozen-Q/trainable-gate
anchors, fully frozen bypass, live sigmoid storage aliases, delayed D2H, policy
fractions 0/0.5/1, below-threshold selection and pre-reset pinned ownership remain.
All seven BF16 native decoder topologies/configurations, seven BF16 graph cases,
four eager low-precision and three graph low-precision comparisons are retained.
#8243's three native GPT cases, #8555's A/B/A mRoPE regression and #8556's layout
and real TE/DDP gradient-dtype cases are unchanged at their earlier exact heads.

## Source identities

| Increment | PR head | Executed #7399 composition |
| --- | --- | --- |
| feature | `6ce197ba0ef81c3467091414b7455417c6269939` | `7d4ed264cacc51eb5db4b63fde3136b0a6fad5d0` |
| gdn2 | `c5284187e9592c8774eca7e01cfd4806dfcc631b` | `cd4f40d653c664a703cf50615ce49525bf3a33fd` |
| graph | `f7b8f8f6fa6030da6129e954ba4dacaac542e5bd` | `de7b56b6acea62724df6c80a845ef55cedf8acf5` |
| precision | `b2a2b61ac306fe59561ed384f84937c5228f0099` | `d774e0b24fa2e4a523f694ae6ea64c8e89475704` |

Compositions add only the manager patch from #7399 at
`aba09a62863cc8ba592da55a4c2fcb28ca0b5b86`. They do not claim to validate the GDN
heads on main without that prerequisite. `source/*/prerequisite.patch` records
the exact delta; `test-only-delta.patch` records each test reduction. Main is
`412eacc96331b9ea622598d9022eb6cb8b66ff85`.

## GPU results

| Run | Device | Cases per rank |
| --- | --- | --- |
| feature-single | RTX A6000 ×1 | 29 passed |
| feature-two-rank | RTX A6000 ×2 | 4 passed: TP2+SP, CP2, PP2, PP2+VPP2 |
| gdn2-single | RTX A6000 ×1 | 10 passed |
| graph-single | RTX A6000 ×1 | 5 passed |
| graph-two-rank | RTX A6000 ×2 | 2 passed: TP2+SP, PP2 |
| precision | RTX PRO 6000 SM120 ×1 | 21 passed |

No selected test is skipped. Native training compares loss, gradients, weights
and complete Adam state. Graph cases require actual replay; positive cases
require transfers; pinned ownership is checked before reset. Bypass guards and
method observers do not replace the numerical kernels, copies or training loop.
Delay injection stays confined to the explicitly named asynchronous stress tests.

Runtime: PyTorch 2.11.0+cu130, TE 2.20.2, FLA 0.5.1 and causal-conv1d 1.6.1.
`results/*/command.json` and launcher sources record exact commands, device UUIDs
and runtime paths. Logs are losslessly compressed. The remote precision source
was checked against all 3553 regular tracked files in `precision-source-sha256.json`.
The NVFP4 launcher disables RHT/stochastic rounding for exact comparisons and
does not use a high-precision backward override or bypass support checks.

Black, isort, Ruff and project Pylint pass on the changed test file in all four
source trees (`results/lint`). No performance or convergence claim is added.
Upstream GPU CI, landing #7399 and NVFP4 qualification on SM100/SM103 remain pending.

The [previous acceptance archive](https://github.com/WEIXILIN0307/Megatron-LM/tree/8753398273941502e25868d85da470b890fed8ce/gdn-scope-20261123)
retains old/new failure controls, unchanged PR results and versioned historical
performance. Those measurements are not performance acceptance for these heads.
No production change is made to #8243, #8555, #8556 or the closed #8244.

Run `python3 -S audit.py` here to verify checksums, source identities and actual
test outcomes. Reproduce the recorded pytest node IDs through torchrun with the
normal project fixtures and TE/FLA dependencies, adapting only machine paths.
