# FSDP v2 partial CUDA Graph: prerequisite split and validation

This record covers two independent correctness fixes and the bounded, ordinary-module
partial-graph integration. It contains no Combined1F1B bridge, EP communication reorder
or TE implementation patch. All source commits are signed and carry DCO trailers.

| Source | Commit |
| --- | --- |
| Starting main | a7a6d2225700a8ee701453288894501d868c6bd4 |
| Evaluation fix | 41103622852c379074ad993cf592fb799df65a21 |
| Native checkpoint fix | 2886280662e31c1678e9ceb5bb4c94a2c09304e9 |
| Graph-only PR | 5c12bd9af86d3ba098eadd6aeaad4d154100f725 |
| Composed validation | 269f02ebd652a4787691c3799f89c3ff232af24e |

The graph-only source excludes both prerequisites. Its acceptance tests and real-text
runs were exercised with both fixes in the composed source. This distinction matters
when reproducing results before the prerequisite PRs merge into main.

## Results

| Run | GPUs | Passed per rank |
| --- | ---: | ---: |
| Independent evaluation fix (`eval-final`) | 2 | 5 |
| Independent native checkpoint fix (`checkpoint`) | 2 | 2 |
| Composed focused regressions (`core-two-clean`) | 2 | 23 |
| Composed ordinary-path EP2/expert-DP2 (`core-four-clean`) | 4 | 2 |

The four-GPU cases use small GPT MoE models and synthetic tokens. Recorded groups are
DP [0,1,2,3], EP [0,1]/[2,3] and expert-DP [0,2]/[1,3], with four distinct GPU UUIDs.
They cover checkpoint continuation/recapture and dropout/RNG.

A 2.09B, 16-layer/16-expert GPT MoE proxy trained on real WikiText-103 through stock
pretrain_gpt.py: BF16, ZeRO3, EP2, TP=PP=CP=1, sequence 1024, GA2, dropout 0.1.
Ordinary eager and graph each completed 12 steps without skips. Every loss matches.
At steps 4 and 12, all logical local parameter, gradient, main-weight and Adam bytes
were hashed; all digests and metadata match on both ranks. Graph forward replay counts
are 32 per step after capture and zero in eager. This is not complete OLMoE or convergence
qualification. No large checkpoint was written. These state-verification runs are not
performance benchmarks.

A separate ordinary EP1/expert-DP2 run traces only iteration 10 and records 64 graph
launches per rank. Graph compute excludes graph-internal NCCL:

| Communication | Rank 0 intersection (ms) | Rank 1 intersection (ms) |
| --- | ---: | ---: |
| FSDP AG, ordinary/expert meshes aggregated | 26.151144 | 27.837883 |
| Expert-DP RS | 29.186084 | 28.261979 |
| Ordinary-DP RS | 0 | 0 |

AG attribution is aggregated because the NVTX prefetch caller can differ from the
actual target. RS uses the reducing unit; no RS is unattributed. These are GPU kernel
interval intersections, not network progress, hidden critical-path time or speedup.
EP1 has no EP all-to-all, so empty EP-pair fields in the generic analyzer are not EP
measurements. The EP2 state runs and EP1 trace use different topologies and are not
compared numerically against one another.

## Reproduction

Use PyTorch 2.11.0+cu130, TE 2.20.2 and pytest 9.1.1. The installed TE was unmodified.
Checkout the corresponding source commit and run torch.distributed.run with 2 or 4
processes. The exact pytest node IDs and environment-specific OCI commands are in each
run's command.json. For example, on the composed source:

```bash
python -m torch.distributed.run --nproc-per-node 2 -m pytest --experimental -q \
  tests/unit_tests/distributed/mfsdp_v2/test_mcore_partial_cuda_graph.py
```

That example selects the entire module; the recorded 23-case run selects representative
node IDs listed in runs/core-two-clean/command.json. The four-card selection is recorded
separately. Real-text arguments are in runs/training-args.json. Adapt the tokenizer,
indexed WikiText and artifact paths to the local environment. The model uses random
initialization and no downloaded weight checkpoint. Entry-point adapters observe
states/graph launches and profile one step; they do not change model computation.

Raw Nsight reports are included. SQLite exports are retained locally and omitted here
to avoid duplicating reports; recreate them with the package version/hash/URL recorded
in runs/nsys-package.json:

```bash
nsys export --type sqlite --output trace-rank-0.sqlite trace-rank-0.nsys-rep
```

Export both ranks into runs/profile-standalone-ep1 and run runs/analyze_profiles.py.
Tiny last-digit float-summation differences across Python versions do not change the
recorded kernel intervals. runs/compare_states.py reproduces the digest comparison.
Source manifests contain every tracked regular-file hash for each tested source tree.

## Failure diagnosis and limits

The first four-card teardown assertion grew by 17,039,360 bytes per capture lifetime.
A diagnostic wrapper cleared only cuBLAS handle/stream workspaces at the existing
measurement point. All three measurements then matched exactly on every rank, and
numerical parity passed. The final test uses the existing test_memory.py cache-control
method, preserves its original assertion and changes no production cache behavior.
Original failures and the diagnostic are retained; only the `*-clean` runs qualify the
final test. Earlier runs belong to the manifests named pretyping/before-cache-fix.

MCore input/finalization recovery passes with stock TE. TE 2.20.2 warmup exceptions may
leave private temporary hooks and need a separate TE fix or process restart. No blanket
TE recovery, EP traffic hiding, stable speedup or full upstream CI result is claimed.
Black/isort/Pylint/Ruff pass against the cached fetched base. Mypy remains non-gating
under the repository script, with existing mixin/stub diagnostics; new diagnostics were
fixed. Original task source and measurements are preserved separately.

Temporary source archives, extracted test trees, profiling tools and caches were
removed only after evidence/source verification. Shared OCI, venv, CUDA, tokenizer,
WikiText and other tasks were preserved. See runs/cleanup*.json.
