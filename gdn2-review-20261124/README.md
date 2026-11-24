# GDN2 offload review follow-up

PR head: `03a7f265b2e8777999dfae0be5e5966e72603a44`. The executed source was
`186d2213f8444884facd8c340c8c4b402c0247d9`, with the manager patch from #7399 applied.
`prerequisite.patch` records that delta.

- RTX A6000 ×1: all four single-rank decoder tests and the three GDN2
  gate-storage tests passed.
- RTX A6000 ×2: TP2+SP, CP2, PP2 and PP2+VPP2 decoder tests passed on both
  ranks. No selected case was skipped.
- The unexpanded trainable-gate test was also run on
  `1de6ef6b83579082e08ca1ccec687249a329a75c` with only the GDN2 gate exclusions
  removed. Forward output and gradient comparisons passed, and the transfer
  assertion failed because the live `w` storage was copied to pinned host memory.
  The directory `review-gate-control` contains the exact local delta and logs.

Each run includes the command, source identity, device selection and per-rank
reports. Use `sha256sum -c SHA256SUMS` to verify the files. The
[earlier 36-case run](https://github.com/WEIXILIN0307/Megatron-LM/tree/046f778ed3b1f9d8d41b02c55021180fd3f99be2/gdn2-focused-final-20261124)
covers the previous PR revision. End-to-end memory and runtime measurements
for GDN2 eager offloading remain pending.
