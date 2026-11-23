# GDN recurrence activation offloading

`gdn_core_attn` offloads tensors saved by the BF16 FLA GDN/GDN2 recurrence
using the existing scheduler, transfer streams and pinned-memory pool. It is
opt-in and trades device memory for host-pinned storage and copy traffic.
Numerical kernels are unchanged.

## Usage

Add these arguments to an existing GDN training launch:

```bash
export NVTE_CPU_OFFLOAD_V1=1
# Append to the training arguments:
--bf16 --experimental-attention-variant gdn \
--fine-grained-activation-offloading --offload-modules gdn_core_attn
```

Programmatic configurations use `fine_grained_activation_offloading=True`,
`offload_modules=["gdn_core_attn"]` and `gdn_kernel_backend="fla"`.
`min_offloaded_tensor_size` is measured in elements (default 1,048,576).
`activation_offload_fraction` selects a fraction from 0 to 1; zero selects no
copies. The existing warmup policy and last-group margin apply. Use multiple
GDN layers when evaluating steady-state savings.

## Supported scope

The scope wraps only the FLA recurrence and commits before output normalization.
A differentiable recurrence input anchors backward prefetch, including a trainable
gate when Q is frozen. Evaluation, no-grad execution and fully frozen recurrences
bypass the scope. Inputs shared with neighboring operations are not force-released.

BF16 GDN, selective `gdn_norm_out` recomputation and native pre-GDR fusion are
supported. Configuration rejects non-BF16 activations, other recurrence backends, full
recomputation and non-TE CUDA graphs. Projection, convolution and output normalization
remain outside this scope. Projection/convolution recomputation and convolution
input offload are separate work in [#7852](https://github.com/NVIDIA/Megatron-LM/pull/7852).

GDN2's `w` gate and unexpanded `b` gate remain resident when trainable: sigmoid
backward outside the recurrence still retains their storage. Expanded or frozen
gates can be offloaded. The recurrence uses the same scheduler and pinned pool.

## Validation

`tests/unit_tests/ssm/test_gated_delta_net_offloading.py` covers fractions,
thresholds, packed sequences, frozen inputs, delayed copies and pool reuse.
`tests/unit_tests/ssm/test_gdn_offload_training.py` compares native decoder training
with loss, gradients, weights and complete optimizer state, including pinned
ownership before schedule reset.

Historical Qwen measurements at `1f2c9ae4f7e617b12caa115e387d95e668517c62`
reduced peak allocated memory by 952 MiB on A6000, with step-time and pinned-memory
costs. They do not qualify the current revision. The [versioned evidence](https://github.com/WEIXILIN0307/Megatron-LM/tree/5af996df67ef2483242ea25be876b996d745449c)
retains commands, original results and experimental profiling/training tools.

## Transformer Engine CUDA graphs

Set `cuda_graph_impl="transformer_engine"`, a positive warmup count and either
attention capture (`cuda_graph_modules=["attn"]`) or whole-layer capture (`[]`).
Local and full-iteration graph modes are rejected for this recurrence scope.
Delayed offload uses the existing `delay_offload_until_cuda_graph` option.
The graph regression compares complete training states after capture/replay and
checks actual transfers and pinned-pool release, including a mixed GDN/attention
model with both recurrence and standard-attention offloading enabled.

Historical graph measurements at `c5264b66197040f4309edfdb22dc9bbaf62189b1`
reduced reserved memory, not peak allocated memory, and increased step time.
They do not establish a current-version performance benefit.

## Low-precision Transformer Engine layers

FP8 and NVFP4 apply to the TE projections and MLPs; the FLA recurrence and its
offloaded saves stay BF16/FP32. FP8 delayed and tensorwise scaling use the
existing recipes. BF16 remains the default.

TE graph capture forwards `disable_parameter_transpose_cache` to
`fp8_weight_caching` for both FP8 and NVFP4. Its default is unchanged; BF16 graph
kwargs are unchanged. NVFP4 graph offloading requires the explicit programmatic
setting `disable_parameter_transpose_cache=True`; configuration rejects the
cached combination. Disabling caching requantizes weights on each microbatch.

NVFP4 results on RTX PRO 6000 (SM120) do not replace qualification on SM100/SM103,
the training devices documented by TE 2.20.2. The separate NVFP4 fused weight-gradient
output alignment and width-4112 dgrad failures are not recurrence-offload fixes.
