# Two-A6000 current-runtime performance comparison

**This is an OLMoE-derived 16-expert proxy, not the complete public OLMoE model.** It preserves 16 attention/MoE blocks, hidden size 2048, expert FFN 1024, and top-8 routing; the expert count is 16 instead of 64. Parameters are randomly initialized and real WikiText-103/tokenizer inputs match the earlier full-model validation. BF16, ZeRO-3, EP2, GA2, sequence 1024, global batch 4, attention/hidden dropout 0.1. Two 48GB A6000 GPUs (5/6) were used. The production runtime remains unchanged at PR head `45bc0d38bf6b87208a6807f013597aadd68b8f54`; source hashes are in `plan.json`.

Both modes keep FSDP parameter-gather and gradient-reduction overlap enabled. The measured ratio evaluates the complete partial-graph configuration without isolating the contribution of a specific kernel or communication mechanism.

Three predeclared paired repeats used order eager-1, partial-1, partial-2, eager-2, eager-3, partial-3. Every run trained 60 optimizer steps. Steps 1–15, including capture and post-capture warmup, are excluded from steady measurements; steps 16–60 give 45 measured steps per run. Profiling, evaluation, and checkpoint writing were disabled equally.

An external wrapper times the stock `train_step`, which includes data consumption, forward/backward, communication, gradient handling and optimizer update. CUDA is synchronized at both boundaries. There is no per-step distributed timing collective. Step summaries use the slower rank; sustained throughput uses the wall-clock span from the earliest rank start of step 16 to the latest rank completion of step 60. This includes between-step logging/measurement overhead but excludes final logging, initialization and capture. Both ranks share the same host monotonic clock.

| Mode | Mean steady wall s/step (repeat SD) | Global tokens/s | Steady allocated / reserved GiB | Sampled GPU GiB | Mean capture s |
|---|---:|---:|---:|---:|---:|
| eager | 1.2057 (0.0169) | 3397.2 | 21.83 / 23.08 | 25.66 | 0.000 |
| partial | 1.1158 (0.0038) | 3670.9 | 20.72 / 24.57 | 27.18 | 2.196 |

Observed steady throughput ratio: **1.0806x**, or **8.06%**. Paired ratios: 1.0928x, 1.0806x, 1.0682x.
All six 60-step loss histories match exactly on both ranks; all runs completed with no skipped steps. Estimated capture amortization: 25 steady steps (capture cost divided by observed per-step saving; not measured full-job time-to-solution).

Peak allocated/reserved values are PyTorch allocator metrics, not total GPU memory. The separate GPU column is the maximum sampled NVML device memory occupancy during each steady window, sampled at one-second resolution. Steady peaks reset before step 16; run peaks include initialization/capture/earlier training peaks observed by the wrapper. Raw per-rank rows retain all memory and capture values.

Limitations: shared host and PCIe topology; three repeats are workload-specific evidence, not a general hardware claim. With EP2 on two GPUs, expert-DP size is one; this timing experiment does not measure expert-DP collective overlap. The full-model four-GPU trace is separate evidence. No full-OLMoE, dedicated EP-overlap, convergence, or other-precision speedup is inferred.

Reproduce the argument lists with `benchmark_entry.py`, setting paths/environment per `environment.json`. Run `python analyze_performance.py` in this directory to recompute the report from published per-rank measurements. The commands and source checksums are recorded; no model weights, text, checkpoint shards, or host credentials are published.
