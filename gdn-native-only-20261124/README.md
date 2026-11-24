# GDN offload validation through normal execution paths

The GDN/GDN2 tests use the existing transfer scheduler. GPT fraction tests
use next-token cross-entropy. The mRoPE replay comparison generates position
IDs with the training helper for text batches with different document boundaries.
Evaluation and freezing assertions observe active chunks.

| Run | Executed source | GPU ranks | Outcome per rank |
| --- | --- | ---: | --- |
| gpt-fractions | `214219f59aaeadaaaaa136347d6a3c2a1a29c5d5` | 1 | 3 passed |
| gdn-single | `94daa1c58d26b636a53f6ae313d4e3de7d10aeb4` | 1 | 25 passed |
| gdn-two-rank | `94daa1c58d26b636a53f6ae313d4e3de7d10aeb4` | 2 | 4 passed |
| gdn2-single | `a6ac96e1b4592afce55f54b67eafeda7fdf4f68c` | 1 | 7 passed |
| graph-single | `c75b6b71405df71a034760d1651e77f88dfd193e` | 1 | 5 passed |
| graph-two-rank | `c75b6b71405df71a034760d1651e77f88dfd193e` | 2 | 2 passed |
| precision | `52bdce700f632ea2bb0ca6595a5046cfcddae566` | 1 | 21 passed |
| mrope-fixed | `e66e9098c4befd3183b3b2b0282da0759198bc56` | 1 | 1 passed |
| mrope-control | `6104746e4c5fde2f9c2e5815a3668cbbf26598ef` | 1 | 1 expected failure |
| alignment-layout | `84534e4372a3ce352728d33cd545c63b6d5fe5c3` | 1 | 33 passed |

All passing selections ran without skips. BF16 GDN/GDN2 runs and the
layout run used RTX A6000. Precision and mRoPE runs used RTX PRO 6000 (SM120).
The mRoPE control is main plus the identical test. The current NVFP4 DDP
layout result is unit coverage. Historical native decoder before/after GPU
results remain attached to their original revisions in the
[training archive](https://github.com/WEIXILIN0307/Megatron-LM/tree/02890e97b1273d9a8695f76aaa208730054fe415/gdn-open-issues-20261027).

Each run records the command, exact source, GPU selection and rank output.
Verify the archive with `sha256sum -c SHA256SUMS`.
