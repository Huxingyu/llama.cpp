# SpacemiT MTP development

This branch extends the existing MTP implementation at upstream commit
`855a53bedbc3fdb6f9409e91ca98c2f4f815a182`. It does not implement a second
speculative decoding algorithm.

## Call flow and backend ownership

`common/speculative.cpp` owns draft generation, target verification and acceptance.
`src/models/qwen35.cpp::graph_mtp` and `qwen35moe.cpp::graph_mtp` describe the MTP
head using ordinary ggml operations. Backend support is decided in
`ggml-spacemit.cpp::ggml_backend_spacemit_device_supports_op` and
`ime.cpp::ggml_spacemit_get_tensor_traits`; actual dispatch passes through
`ggml_spacemit_compute_opnode`, the traits implementation and RVV/IME/scalar kernels.

SpacemiT is registered as an accelerator rather than a GPU. Automatic GPU
selection therefore does not place the model's KV cache there. For full backend
coverage, select `--device SPACEMIT0 -ngl 99` explicitly. The input embedding table
is CPU-resident by default in llama.cpp; the standard override
`--override-tensor '^token_embd\.weight$=SPACEMIT'` places it on this backend.
Neither setting changes the device type or the sampling algorithm.

## Changes

| Change | Reason and implementation |
| --- | --- |
| Stream-scoped cooperative pair barriers | Native spin waits could starve tiles after runtime grid synchronization. Dense and expert paths now use barriers owned by their launch stream; an unmatched final worker never waits. |
| Expert routing prefix ranges | Duplicate expert IDs can require more than one mapping per token for the same expert. Counts and prefix offsets give each expert a disjoint range; workspace is proportional to actual assignments plus expert count. |
| Canonical buffer interface | Public buffer reads and CPU copies must expose ggml bytes, rather than physical packed blocks. Q4_0 is inverted a block at a time without a second model copy. Other repacks retain original bytes in the same accounted allocation. Partial writes and memset rebuild the packed representation. Those less common layouts have an explicit memory cost. |
| SUM | Row-contiguous F32 reductions with arbitrary outer strides accumulate in FP64. Per-worker partials use separate cache lines and a runtime grid barrier. This enables complete cache-snapshot graphs instead of silently skipping them. |
| GDN on IME1 | The RVV implementation requires the state row to fit in an m8 vector. Dispatch checks the actual vector capacity and uses the existing scalar reference for larger rows, including 128-wide K1 heads. |
| Embedding GET_ROWS | F16, BF16 and quantized rows dequantize to F32. Packed weights read only the requested canonical row. This covers the pinned model's Q6_K embedding as well as Q4 formats. |
| TOP_K | Partial sorting follows the CPU implementation, including ties, infinities and its final output swap, so backend sampler graphs do not need this CPU fallback. |
| M2/M3 quantization candidate | `GGML_SPACEMIT_SMALL_M_QUANT=1` partitions independent row/K blocks across workers. M4 retains its interleaved four-row quantizer. The candidate is disabled by default pending board measurements. |
| Speculative example limits | Count actual emissions including EOG, stop at zero/exhausted budgets, bypass the speculator's special zero-draft default when only one token remains, and propagate decode errors. |

## Diagnostics

`GGML_EXEC_TRACE=/path/nodes.jsonl` records context roles, shapes/strides/types,
backend placement, tensor flags, whether a node actually requires computation,
completed graph execution, actual backend dispatch, fused members and selected
IME layouts. Empty tensors and nodes without `GGML_TENSOR_FLAG_COMPUTE` remain
visible in placement records but do not require a kernel dispatch. Placement and
execution are separate events.
Diagnostic mode synchronizes asynchronous backends; **do not use it to measure
performance**. With the variable unset, the normal asynchronous entry point is
used unchanged.

`LLAMA_MTP_TRACE=/path/tokens.jsonl` records draft, verification, checkpoint
restore and actual emitted-token events in completion/speculative-simple.
Verification suffixes after EOG are not counted as emitted output. Server response
`return_tokens` and `timings` provide the independent HTTP integration evidence.

## Focused checks

Run `test-backend-ops -b SPACEMIT0` with one of the environment selectors:

| Selector | Operation filter | Coverage |
| --- | --- | --- |
| `OSPP_MTP_SMALL_MATMUL=1` | `MUL_MAT` | Small draft/verification matrices; `OSPP_MTP_EXTENDED=1` adds output-column boundaries. |
| `OSPP_MTP_MOE=1` | `MUL_MAT_ID` | 576 expert-routing cases, ordinary and duplicate IDs, broadcast activations, M1–M4. |
| `OSPP_MTP_STATE=1` | `SUM\|GATED_DELTA_NET_CACHE_FUSION\|CPY` | 42 reduction, GDN snapshot and strided-copy graphs. |
| `OSPP_MTP_GATHER=1` | `GET_ROWS` | 252 embedding cases, including Q6_K. |
| `OSPP_MTP_SELECTION=1` | `TOP_K\|SET_ROWS` | 24 sampler/KV update cases, ties and negative infinity. |

Focused runs require every selected case to execute; zero execution or unsupported
cases fail. `OSPP_CANONICAL_INPUTS=1` uses CPU-initialized reference data; set it to
`0` to exercise the target-to-CPU canonical-copy direction. Do not change numerical
tolerances to make these checks pass.

`test-spacemit-buffer` checks 30 format/shape combinations through public set,
get, partial-range, memset and cross-backend copy APIs. It also works with IME2
packing selected, without invoking IME2 matrix instructions.

## Remaining hardware and model scope

The functional model is a pinned third-party Unsloth GGUF of Qwen3.5-0.8B with an
MTP head. Dense-model inference and synthetic expert routing are distinct evidence;
this does not establish whole-model Qwen MoE behavior. IME2 matrix execution and
performance require a compatible board. QEMU does not provide board throughput,
latency or optimization speedup evidence. Upstream pull requests are a separate
publication step.
