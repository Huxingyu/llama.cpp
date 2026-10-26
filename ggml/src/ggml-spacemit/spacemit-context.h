#pragma once

#include "ggml.h"

#include <spert.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

struct spacemit_opnode;  // forward declaration (defined in spacemit-opnode.h)

namespace ggml::spacemit {

inline constexpr size_t cache_line_size_f32 = 64 / sizeof(float);

struct context {
    spert::Context *        runtime       = nullptr;
    spert::Barrier *        pair_barrier  = nullptr;
    uint32_t                ith           = 0;
    uint32_t                nth           = 0;
    void *                  workspace     = nullptr;
    size_t                  workspace_size = 0;
    spert::SharedBufferView shared        = {};

    void reset(spert::Context & runtime_in, uint32_t ith_in, uint32_t nth_in,
               void * workspace_in, size_t workspace_size_in, spert::SharedBufferView shared_in,
               spert::Barrier * pair_barrier_in) {
        runtime        = &runtime_in;
        pair_barrier   = pair_barrier_in;
        ith            = ith_in;
        nth            = nth_in;
        workspace      = workspace_in;
        workspace_size = workspace_size_in;
        shared         = shared_in;
    }

    void clear() {
        runtime        = nullptr;
        pair_barrier   = nullptr;
        ith            = 0;
        nth            = 0;
        workspace      = nullptr;
        workspace_size = 0;
        shared         = {};
    }

    void sync() { runtime->sync(); }

    // A raw spin wait can prevent another ready tile from resuming after
    // a cooperative grid sync. Keep paired waits in the same runtime domain.
    void pair_sync() {
        if (pair_barrier == nullptr || runtime->barrier(*pair_barrier) != spert::Status::Ok) {
            throw std::runtime_error("ggml-spacemit: pair barrier failed");
        }
    }
};

class tensor_traits_base {
  public:
    virtual ~tensor_traits_base() = default;

    virtual bool work_size(int n_threads, const ggml_tensor * op, size_t & size) const = 0;
    virtual bool compute_forward(context & ctx, ggml_tensor * op) const                 = 0;
    // Fused multi-op dispatch, keyed on spacemit_op_code. Default is "not supported".
    virtual bool compute_forward_fused(context & ctx, const spacemit_opnode & node) const {
        GGML_UNUSED(ctx);
        GGML_UNUSED(node);
        return false;
    }
    // Layouts with a lossless inverse need no canonical shadow allocation.
    virtual bool can_unpack() const { return false; }
    virtual void unpack(const ggml_tensor *, void *, size_t, size_t) const {
        throw std::runtime_error("ggml-spacemit: layout has no lossless inverse");
    }
    virtual int  repack(ggml_tensor * tensor, const void * data, size_t size) const     = 0;
};

}  // namespace ggml::spacemit
