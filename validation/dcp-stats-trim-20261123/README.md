# DCP summary validation

Tested source `5623577007cd87e1fc4377b9851c2f96d4d0dad2`, on dev `6559f64c2`.
The documented command ran five BF16 updates on two A6000 GPUs. The focused
CPU selection passed 15 cases, and the four-rank PP2×CP2/VPP2 scheduler off/on
test passed on each rank. A four-A6000 TP2×CP2 checkpoint replay matched
schedule, legacy FLOPs fields and inputs, losses, gradients and parameters
bitwise on all five resumed updates. The empty disabled path, evaluation
exclusion and the enabled stdout/TensorBoard windows at steps 2, 4 and 6
were checked. No benchmark or live W&B run was performed.

The container and commands use the same Python 3.12.4, Torch 2.11+cu130,
TE 2.20.2 and FA2 2.8.3 stack as the November 21 acceptance. Adapt paths
and GPU indices in the recorded commands for replay. Use `run_training.py`
on the host and `verify_training.py` from the source checkout in the
container. Checkpoints are identified by hash; binaries are omitted.

CPU: `python -m pytest -q tests/unit_tests/test_sequence_packing.py -k 'dcp_schedule_stats and not distributed'`
with CUDA hidden. Distributed: `torchrun --standalone --nproc-per-node=4 -m pytest -q tests/unit_tests/test_sequence_packing.py::test_dcp_schedule_stats_distributed`.
