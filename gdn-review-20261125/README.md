# GDN offload and mRoPE checks

| PR | Run | Source commit | Result |
| --- | --- | --- | --- |
| #7908 | [gdn-eager](gdn-eager/) | `c94c526106274945c6f206329cdcfe10780879c0` | RTX A6000: 22 passed, 4 multi-GPU cases skipped |
| #9712 | [gdn2-eager](gdn2-eager/) | `f0302fb902786b00aa4290252bbba308a6d01fc2` | RTX PRO 6000: 30 passed, 4 multi-GPU cases skipped |
| #8308 | [graph-single](graph-single/) | `7ec7bda873319ce88d095b563a1073544e9edca5` | RTX PRO 6000: 5 passed |
| #8308 | [graph-two-rank](graph-two-rank/) | `7ec7bda873319ce88d095b563a1073544e9edca5` | RTX PRO 6000: TP2+SP and PP2 passed on both ranks |
| #8308 | [four-rank-initial](four-rank-initial/) | `7ec7bda873319ce88d095b563a1073544e9edca5` | RTX PRO 6000: first 3 cases passed on all ranks, run stopped during fourth |
| #8308 | [four-rank-eager-gdn2](four-rank-eager-gdn2/) | `7ec7bda873319ce88d095b563a1073544e9edca5` | RTX PRO 6000: GDN2 eager passed on all ranks |
| #8308 | [four-rank-graph-gdn2](four-rank-graph-gdn2/) | `7ec7bda873319ce88d095b563a1073544e9edca5` | RTX PRO 6000: GDN2 whole-layer graph passed on all ranks |
| #8543 | [precision](precision/) | `b5c1b13934c985f02bf39b33f06f93e5b1e426cf` | RTX PRO 6000 SM120: 22 passed |
| #8543 | [precision-with-alignment](precision-with-alignment/) | `2927b7c1565a97d3bccc712f0db806b3a40c651d` | With #8556: NVFP4 fused wgrad TP2+SP passed on both ranks |
| #8555 | [mrope-packed-control](mrope-packed-control/) | `e66e9098c4befd3183b3b2b0282da0759198bc56` | Expected mismatch (16 vs. 32 positions) on both CP ranks |
| #8555 | [mrope-packed-fixed](mrope-packed-fixed/) | `80f01f6c3cfc834c891e234ecc8939d9934b7306` | Packed CP2 rotary registration passed on both ranks |
| #8555 | [mrope-text](mrope-text/) | `80f01f6c3cfc834c891e234ecc8939d9934b7306` | Native text A/B/A replay passed |
| #8555 | [mrope-packed-replay](mrope-packed-replay/) | `80f01f6c3cfc834c891e234ecc8939d9934b7306` | Packed CP2 replay passed on both ranks with supported TE backward |

Commands preserve the arguments with machine-local paths and GPU UUIDs redacted. Rank JSON records pytest outcomes for completed launches. Use `gzip -dc` to read the rank stdout/stderr logs. Verify the archive with `sha256sum -c SHA256SUMS`.

The BF16 four-rank [summary](four-rank-summary.json) covers TP2+SP+PP2 on a small 8-layer native decoder with sequence length 128. The first launch was stopped during its fourth case. The per-rank stdout matches the first three passes, and the isolated GDN2 reruns exit successfully. The two- and four-rank graph runs used `NCCL_P2P_DISABLE=1`. These are correctness checks on shared GPUs, not full-Qwen timing measurements.

NVFP4 runs disabled RHT and stochastic rounding. The standalone precision runs used unfused weight gradients. The separate fused-wgrad check composes the published precision commit with the [alignment patch](nvfp4-alignment-on-precision.patch). The packed mRoPE end-to-end probe changes only the test and uses a supported non-deterministic TE THD backward on SM120.
