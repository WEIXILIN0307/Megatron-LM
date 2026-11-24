# GDN offload regression checks

| PR | Run | Executed source | GPU ranks | Result per rank |
| --- | --- | --- | ---: | --- |
| #7908 | [gdn-single](gdn-single/) | `711e63a982bfa2da1a5a4cbe9d244106dcab1e86` | 1 | 15 passed |
| #7908 | [gdn-two-rank](gdn-two-rank/) | `711e63a982bfa2da1a5a4cbe9d244106dcab1e86` | 2 | 4 passed |
| #9712 | [gdn2-final](gdn2-final/) | `47239378149451d68b8f1e825c18f2c22ac50e9d` | 1 | 7 passed |
| #8308 | [graph-single](graph-single/) | `ef090f9433687439d72c995dee4e07b297f1b3f9` | 1 | 5 passed |
| #8308 | [graph-two-rank](graph-two-rank/) | `ef090f9433687439d72c995dee4e07b297f1b3f9` | 2 | 2 passed |
| #8543 | [precision](precision/) | `a057e7b9f2b41f86d6f5d2b2015ed45dcf5f8137` | 1 | 14 passed |

All selected cases ran without skips. BF16 runs used RTX A6000. FP8 and NVFP4 used RTX PRO 6000 (SM120). NVFP4 used unfused weight-gradient accumulation with RHT and stochastic rounding disabled.

The native decoder cases compare loss, gradients, weights, and optimizer state. The graph cases also check replay and host transfers. Each run records the command, source, environment and logs for each rank. Verify the files with `sha256sum -c SHA256SUMS`.

[Earlier regression logs](../gdn-native-only-20261124) are pinned to their original source revisions. [Earlier test sources](https://github.com/WEIXILIN0307/Megatron-LM/tree/52bdce700f632ea2bb0ca6595a5046cfcddae566).
