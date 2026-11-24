# Focused GDN2 offload tests

#9712 reuses the existing GDN recurrence replay test for GDN2's resident and
partial packed/norm cases. Outputs, all gradients, warmup, repeated pool reuse
and pre-reset ownership checks remain; GDN2 now also gets the shared chunk-policy
and copy-selection assertions. A config-field readback is removed because real
GDN2 model construction already validates acceptance.

The gate-storage test keeps three distinct behaviors: trainable unexpanded gates,
trainable expanded gates and frozen gates. The second frozen-head layout follows
the same no-grad ownership path and is removed. Both normal and frozen-query
delayed-D2H cases, evaluation with backward, nonuniform packed `[0,32,128]`,
fractions 0/0.5 and full-offload native decoder/DDP/Adam training remain.
The GDN2 selection is reduced from ten to nine actual GPU cases.

Only `tests/unit_tests/ssm/test_gated_delta_net_offloading.py` changes from the
previous PR heads. #8308/#8543 are restacked to inherit that cleanup; their
production code and their graph/low-precision tests are unchanged. #7908 and
the independent fixes are unchanged.

| Increment | PR source | Executed #7399 composition |
| --- | --- | --- |
| gdn2 | `d7cb8bd77530d0db375163d23bea097e0e914309` | `3fba560996243f902589adc4c095eba4ec978074` |
| graph | `18a3b427515b07766000ef4c2d9d646230cd0cd4` | `fa5178043d93351bb38f1fe114383a688c9e3619` |
| precision | `7f46564b108137222a2da0f6d3fb372ebf579cc1` | `975b17bfecf1f01e0a153e9bf816252843c01d67` |

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
pending. [Earlier evidence](https://github.com/WEIXILIN0307/Megatron-LM/tree/a72b43d16397d5daf91890e75304d213a26ad992/gdn-test-trim-20261123)
keeps its exact source identities and historical measurements.

Run `python3 -S audit.py` here to verify archive integrity and actual outcomes.
Reproduce the recorded node IDs through torchrun with normal project fixtures,
TE/FLA dependencies and the recorded prerequisite, adapting machine paths only.
