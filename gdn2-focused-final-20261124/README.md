# Focused GDN2 delayed-copy coverage

#9712 keeps the delayed D2H case with frozen Q because it exercises the
alternative differentiable anchor. The unfrozen-Q delay is covered by the
GDN case through the same recurrence wrapper and transfer scheduler.
The GDN2 gate-storage tests still compare outputs and gradients. Native
decoder training still compares the complete optimizer state.

- PR source: `55a0035b0a96149d8b94ec510626bf5b6f943308`
- Executed source: `1de6ef6b83579082e08ca1ccec687249a329a75c`
- Prerequisite: the offload manager patch from #7399 at
  `aba09a62863cc8ba592da55a4c2fcb28ca0b5b86`
- RTX A6000 result: 36 passed, including eight GDN2 cases, with no skips

`command.json` contains the exact torchrun invocation. The container launcher
and rank-reporting scripts come from the
[prior GPU archive](https://github.com/WEIXILIN0307/Megatron-LM/tree/9a9431f6ba49198dcba1448654ebac520cda0c6c/gdn2-test-observer-20261124).
`rank-0.json` records the executed node IDs. The two retained GDN delayed-copy
parameters and the frozen-Q GDN2 parameter have explicit names in this revision.
Use `sha256sum -c SHA256SUMS` to verify the archived files.
