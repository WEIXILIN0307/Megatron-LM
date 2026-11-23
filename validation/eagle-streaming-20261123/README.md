# EAGLE3 streaming finetune validation

Megatron-LM runtime `1b369864d4e77a54961dd1a82013b8498853224c` and
documentation head `f319f96c099e7c0a4c1106ffdf4a077aa46d9a8e`.
ModelOpt `0e6a792cfccf0bda1ebaecfec8f9b8d294858d62` on both sides.

The CPU file passed 30 cases, including token and mask shift, BF16 feature
row alignment and the streaming-only forward flag across streaming, disk
and ordinary SFT fixtures.

Live BF16 tests used three A6000s (one producer, one or two trainer GPUs),
vLLM 0.29.0, NIXL 1.5.0 with UCX, CUDA 13 on both sides, PyTorch
2.13.0+cu130 for serving and 2.11.0+cu130 with TE 2.20.2 for training.
The teacher is a seeded, four-layer Qwen3 with hidden size 128, slot
capacity 64 and the Qwen3 tokenizer. The offline checkpoint contains
the matching teacher embedding and final norm. Each TP run completed
two updates, three validation passes of one microbatch and saved step 2.
Each run issued five successful HTTP requests and acknowledgements
after NIXL feature transfer. Requests were sequential, with one TP source
per run.

The TP2 HTTP timeout probe delays a local HTTP response and configures a
one-second read timeout. It reaches the actual finetune batch path,
ModelOpt adapter and peer NCCL broadcast. A second test runs the complete
finetune launcher with the valid checkpoint and a refused endpoint.
In both cases torchrun exits nonzero and terminates the peer after the
source fails. The timeout probe does not exercise a live producer or
NIXL transfer. The fault checks cover the observed torchrun envelope.

Earlier Qwen3-0.6B live runs belong to the original PR head. The full
model download was incomplete in this environment, so the live checks
here use the small checkpoint above. Recovery, concurrent producer
requests and large-scale performance were outside these checks.
An attempted mixed-CUDA transfer failed when the receiver could not
deserialize a descriptor from the other NIXL backend module. The
successful run used CUDA 13 builds on both sides.

The containers and commands record exact runtime paths. Adapt the paths,
GPU UUIDs and prepared package overlays for replay. Generate the teacher
and matching offline checkpoint with the two `make_tiny_*.py` scripts,
start the producer with `run_producer.sh` and use `launch_live.py` for the
trainer pair. The Qwen3 tokenizer comes from the public Qwen3-0.6B
revision `c1899de289a04d12100db370d81485cdf75e47ca`.
`verify_evidence.py` checks fresh replay results with local checkpoint
binaries. The published record retains their hashes and iteration
trackers.
