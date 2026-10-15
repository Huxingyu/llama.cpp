# K3 decode state kernels

Optimize CPY, GET_ROWS, and GATED_DELTA_NET in the standalone SpacemiT backend for Qwen3.5-0.8B decoding on eight A100 workers. Preserve graph/cache semantics, tensor results, non-target fallback behavior, and model quantization. The user approved attempting these three optimizations on a new branch following the measured analysis.

The measured non-MulMat hotspots are CPY 3.33 ms/token (GDN state copies and a strided convolution-state view), GDN 2.64 ms/token, and GET_ROWS 2.03 ms/token. First optimize kernels instead of changing cache ownership or graph aliasing.

CPY: retain conservative overlap handling. Add a vector path for the disjoint three-float rows used by the convolution-state view, with bounded tail loads. Ordinary disjoint contiguous state copies may use a separately tuned normal-memory copy helper.

GET_ROWS: use the normal-memory copy helper for supported element types, retain row/index/stride semantics and handle threads with no work without a negative byte count.

GDN: inspect register pressure in the fused single-token kernel, try a smaller RVV register grouping for heads that fit, and preserve the existing path for other shapes. Keep the existing arithmetic ordering where possible. Validate scalar and vector gates, multiple sequences, tails, and prefill.

Correctness is required before performance acceptance. Reuse test-backend-ops CPU references and add target-layout cases. Compare deterministic full-model outputs/logits where available. Run the existing eight-worker tg128 benchmark on K3 with unchanged clocks and settings, interleaving baseline and candidate. Retain only changes with reproducible benefit; record unsuccessful experiments outside the source tree.
