# GDN2 eager offload validation

PR #9712 at `8026ee8840c09445d3d191394abf8456c9fafb4e` was tested on one RTX A6000.
The seven selected cases cover the native decoder training comparison, three
gate-storage cases, evaluation with gradients, fraction zero, and partial
offload with nonuniform packed sequences and norm recomputation. They use the
existing CUDA streams and scheduling. The run uses the PR source directly.

`command.json` records the exact command. `source.json` records the source
commit and worktree state. `rank-0.json` records the executed node IDs and
outcomes. Use `sha256sum -c SHA256SUMS` to verify the archived files.
