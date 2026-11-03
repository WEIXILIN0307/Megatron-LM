# DCP statistics rebase validation — 2026-11-03

Metrics source: `79bf7b6bcaa95538d2ad7068dc46b7a931fd2c7d` on dev `7b66fcc088e6e60b87c350e0e185721e0373d0ab` (PR #8431).

No compatibility patch was applied. This dev baseline already handles the aligned eager THD padding metadata. The previous A6000 benchmark on metrics head `87d6973da` remains a historical record; **no benchmark or observation-cost measurements were rerun for this rebase**.

## Results

- Four-rank targeted regression: 134 passed per rank, 31 deselected. The filter omits existing eight-rank topology cases and four unrelated optional-import model-builder cases. The full suite was not run.
- Repository Black/isort/Pylint/Ruff checks passed. Optional mypy reported 12 diagnostics identical to the extracted upstream baseline; the diagnostics and baseline source hashes are retained below.
- CP2 on two GPUs, TP2×CP2 on four GPUs, and PP2×CP2/VPP2 on four GPUs: five Adam updates per metrics off/on pair matched bitwise in final schedules, scheduled microbatch counts, loss, gradient hashes and updated parameter hashes.
- Each pair restored identical topology-specific checkpoint, RNG, seed and data inputs. DCP V2 rejects `--deterministic-mode`; successful runs omit this flag, while retaining seed 1234, `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Observed equality does not guarantee determinism for other workloads.
- VPP alignment changed 3 microbatches to 4 and CP1/CP2 sample counts from 7/1 to 3/5. All topologies have two DP×CP ranks and estimated attention balance 0.92.
- The upstream `run_train_loop_without_updates` path ran three loop iterations on two GPUs with metrics enabled: the data hook ran each iteration, no optimizer was built, train_step/statistics computation were never called, parameters were unchanged, and no DCP stdout or TensorBoard metrics were emitted. See `no-updates-check.json` and `train_no_updates.py`.
- Exactly five training stdout records and five TensorBoard events per metric. TP peers receive identical summaries; interleaved evaluation does not contribute. W&B service integration was not exercised.

These are real small dense-GPT BF16 training runs using synthetic tokens, not production convergence or scaling evidence. They use the same model and aligned lengths as the previous acceptance: four layers, hidden 128, FFN 512, four heads, vocabulary 256, sequence limit 1024 and zero dropout. `environment.json` records the installed versions. Internally padded Ampere THD and CUDA Graph execution are unverified.

## Reproduction

`command.sh` in each run directory records the exact launch command; `container-command.txt` records mounts and GPU visibility. Adapt host paths and GPU UUIDs to the local environment. Checkpoints are not committed; their hashes are in `checkpoint-hashes.json`.

In a fresh output directory, copy `launch_train.py`, `train_audit.py`, `lengths.csv` and `verify_acceptance.py`. Set `DCP_REPO_ROOT` to the metrics checkout and `DCP_GPU_INDICES` to the GPUs to use. Select the container with `--container`.

1. Create topology-specific checkpoints with `initial-aligned --iters 1 --save`, `initial-tp --gpus 4 --tp 2 --iters 1 --save`, and `initial-pp-v2 --gpus 4 --pp 2 --vpp --iters 1 --save`.
2. Resume off/on pairs as recorded in `run_correctness.py`. Each uses `--audit --eval --save`, loading its topology-specific checkpoint, with `--on` only in the enabled arm. Each resumes iteration 1 through iteration 6, yielding five updates.
3. Run `python /path/to/verify_acceptance.py` from the metrics checkout. It checks the recorded source hashes against that checkout, off/on equality, per-microbatch work conservation, VPP/CP counts, TP summaries and stdout/TensorBoard ownership. It regenerates `correctness.json`.
4. Run `launch_train.py no-updates --audit --no-updates --on --iters 3 --container <container>` to exercise the real loop without optimizer updates. The script asserts empty summaries/windows and unchanged parameters.
5. Run `unit-command.txt` through the four-GPU container for focused regressions. Run `BASE_REF=dev CHECK_ONLY=true SKIP_DOCS=false bash tools/autoformat.sh` for repository formatting.

The successful commands call real `pretrain_gpt.py`; audit hooks only record results and never modify scheduler decisions. Audit/tensor hashing overhead makes these runs unsuitable for timing claims.
