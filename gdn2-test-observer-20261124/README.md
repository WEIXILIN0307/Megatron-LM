# GDN2 gate-copy observer cleanup

All nine GDN2 behavior cases and their parameters are unchanged. Only the body of
`test_gdn2_offload_retains_live_sigmoid_gates` changes: run the baseline before
installing observers, record each recurrence's gate pointers and copied storage
pointers together, and compare pointer membership directly. This removes the
shared mutable pointer map, the baseline-only guard and repeated name filtering.
The three gate-ownership outcomes and per-layer copy observations remain.

Normal/frozen-Q delayed D2H, evaluation with backward, fraction zero, nonuniform
packed sequences with norm recomputation, pool ownership and full decoder/DDP/Adam
state comparisons all remain. No test is deleted or skipped. AST comparison
confirms that every other test function and all parameter lists are unchanged.

Only `tests/unit_tests/ssm/test_gated_delta_net_offloading.py` changes relative to
the prior heads. #8308/#8543 inherit the same cleanup; production code and their
graph/low-precision tests remain unchanged. #7908 and independent fixes are untouched.

| Increment | PR source | Executed #7399 composition |
| --- | --- | --- |
| gdn2 | `9d45136ace1be5279017e7e17604286b22c0528e` | `3106ae8697a489e98c4c1cb5ed2cc42110cff055` |
| graph | `62c225fff4529dee19999ab38ae87889e18bb47b` | `c45e24c03291d05168f013c9c3bab5a02e344746` |
| precision | `88694d99e81b815e1184de4f2c2c6efa2d7c455d` | `85d68298c0b138d5c3e07c35e384d87831678eab` |

The source base remains `412eacc96331b9ea622598d9022eb6cb8b66ff85`. Validation adds
only the manager patch from upstream #7399 at
`aba09a62863cc8ba592da55a4c2fcb28ca0b5b86`; the feature PRs do not duplicate it.
`source/*/prerequisite.patch` and `test-only-delta.patch` record both exact deltas.
Composed results do not establish passing the GDN heads without #7399.

| Run | Hardware | Executed cases per rank |
| --- | --- | --- |
| gdn2-all | RTX A6000 ×1 | 37 passed, including all nine GDN2 cases and existing shared GDN regressions |
| graph-single | RTX A6000 ×1 | 5 passed |
| graph-two-rank | RTX A6000 ×2 | 2 passed: TP2+SP and PP2 |
| precision | RTX PRO 6000 SM120 ×1 | 21 passed |

No selected test is skipped. Full native decoder tests compare losses, gradients,
updated weights and complete optimizer state. Graph tests require real replay,
positive offload cases require transfers and pinned ownership is checked before
reset. Observers call the original methods; only named stress tests inject copy
delays. Kernels, copies, autograd and optimizer steps are real.

Commands, UUIDs, source identities and per-rank stdout/stderr are in `results/`;
logs are losslessly compressed. The precision source matches all 3553 regular
tracked files in `precision-source-sha256.json`. Runtime: PyTorch 2.11.0+cu130,
TE 2.20.2, FLA 0.5.1, causal-conv1d 1.6.1. Exact NVFP4 comparisons disable RHT and
stochastic rounding before process launch, without a high-precision backward
override or a support-check bypass. Black/isort/Ruff/Pylint pass on the changed
test in all three trees. No new performance or convergence claim is made.

Upstream GPU CI, landing #7399 and NVFP4 qualification on SM100/SM103 remain
pending. [Earlier evidence](https://github.com/WEIXILIN0307/Megatron-LM/tree/6fe467823d1abedb5346502cd3f94d2995f20f9e/gdn2-test-trim-20261124)
keeps its exact source identities and historical measurements.

Run `python3 -S audit.py` here to verify archive integrity and actual outcomes.
Reproduce the recorded node IDs through torchrun with normal project fixtures,
TE/FLA dependencies and the recorded prerequisite, adapting machine paths only.
