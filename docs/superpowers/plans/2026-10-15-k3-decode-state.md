# K3 Decode State Optimization Implementation Plan

> **For agentic workers:** Use subagent-driven-development for bounded kernel implementation and requesting-code-review for independent review. The primary agent owns all K3 runs to avoid benchmark contention.

**Goal:** Reduce eight-worker Qwen3.5 decode latency while preserving the three target operators' results.

**Architecture:** Optimize the standalone backend kernels locally. Use explicit shape/overlap guards and retain generic paths. Graph/cache ownership is unchanged.

**Tech Stack:** C++17, RVV intrinsics, SpacemiT GCC 15.2, spine-runtime 0.6.3, test-backend-ops, llama-bench.

- [ ] Baseline and target regression cases in `tests/test-backend-ops.cpp`: cover f32 CPY three-column strided views, disjoint contiguous state, views/aliases; GET_ROWS one-row large states, short/empty per-worker chunks, multiple rows and views; GDN head sizes 128/256 and existing KDA, sequences, snapshot cases. Compile and run the relevant baseline tests before editing kernels; performance comparison supplies the failing performance requirement for otherwise correct operators.
- [ ] GDN register-pressure experiment in `ggml/src/ggml-spacemit/rvv_kernels.cpp`: inspect generated assembly, choose a bounded fused-kernel path, compare against CPU reference and baseline microbenchmarks.
- [ ] CPY optimization in the same file: exact f32 copies with non-overlap guards and vector tails; verify target and fallback cases before benchmarking.
- [ ] GET_ROWS normal-memory copy experiment: keep the TCM/MulMat memcpy implementation untouched, test element types and partial-worker cases, benchmark 1 MiB state rows.
- [ ] Cross-compile candidate using the same options as `artifacts/build-k3.sh` with `LLAMA_BUILD_TESTS=ON`; build `test-backend-ops`, `llama-bench`, `llama-completion`, and `llama-perplexity` as needed.
- [ ] Run `test-backend-ops test -b SPACEMIT0 -o CPY,GET_ROWS,GATED_DELTA_NET` on K3 with `GGML_SPACEMIT_WORKERS=8` and CPU IDs 8-15. Use targeted parameter filters for baseline failures and compare any pre-existing failure against unchanged source.
- [ ] Run identical full-model inference and benchmark settings: `-dev SPACEMIT0 -ngl 99 -t 8 -C 0xff --cpu-strict 1 -p 0 -n 128 -b 512 -ub 128 -fa on -lm none -r 5 -o json`. Interleave unchanged baseline and candidate; collect operator-level measurements and generation checks. Run pp128 as a prefill regression check.
- [ ] Perform independent spec then code review; resolve findings, verify final diff/tests, commit accepted changes on `perf/k3-decode-state`, and leave the branch unmerged for the user.
