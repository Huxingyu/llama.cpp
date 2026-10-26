#pragma once

// Opt-in diagnostic trace. Completion tracing synchronizes async backends and
// must never be used for performance measurements. Each event is one JSON line.
#include "ggml.h"
#include "ggml-impl.h"
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>

inline bool ggml_exec_trace_enabled() {
    static const bool enabled = [] { const char * path = std::getenv("GGML_EXEC_TRACE"); return path && *path; }();
    return enabled;
}

inline std::string ggml_exec_trace_quote(const char * value) {
    std::string out = "\"";
    for (const unsigned char * p = reinterpret_cast<const unsigned char *>(value); *p; ++p) {
        if (*p == '"' || *p == '\\') { out += '\\'; out += char(*p); }
        else if (*p < 32) {
            char code[7]; std::snprintf(code, sizeof(code), "\\u%04x", *p); out += code;
        } else { out += char(*p); }
    }
    return out + '"';
}

inline void ggml_exec_trace(const char * phase, const char * backend, const ggml_tensor * tensor,
                            const char * kernel = "", int64_t elapsed_us = 0) {
    if (!ggml_exec_trace_enabled()) { return; }
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    static FILE * file = [] {
        FILE * result = std::fopen(std::getenv("GGML_EXEC_TRACE"), "a");
        if (!result) { std::perror("GGML_EXEC_TRACE"); std::abort(); }
        return result;
    }();
    const auto describe = [](const ggml_tensor * t) {
        return std::string("{\"name\":") + ggml_exec_trace_quote(t->name) +
            ",\"flags\":" + std::to_string(t->flags) +
            ",\"compute_required\":" +
            (std::string(!ggml_op_is_empty(t->op) && !ggml_is_empty(t) && (t->flags & GGML_TENSOR_FLAG_COMPUTE) ? "true" : "false")) +
            ",\"type\":" + ggml_exec_trace_quote(ggml_type_name(t->type)) +
            ",\"ne\":[" + std::to_string(t->ne[0]) + "," + std::to_string(t->ne[1]) + "," +
            std::to_string(t->ne[2]) + "," + std::to_string(t->ne[3]) +
            "],\"nb\":[" + std::to_string(t->nb[0]) + "," + std::to_string(t->nb[1]) + "," +
            std::to_string(t->nb[2]) + "," + std::to_string(t->nb[3]) + "]}";
    };
    std::string line = "{\"phase\":" + ggml_exec_trace_quote(phase) + ",\"backend\":" + ggml_exec_trace_quote(backend) +
        ",\"op\":" + ggml_exec_trace_quote(ggml_op_desc(tensor)) + ",\"tensor\":" + describe(tensor) +
        ",\"kernel\":" + ggml_exec_trace_quote(kernel) + ",\"elapsed_us\":" + std::to_string(elapsed_us) + ",\"src\":[";
    bool first = true;
    for (const auto * src : tensor->src) {
        if (!src) { continue; }
        if (!first) { line += ','; }
        first = false;
        line += describe(src);
    }
    line += "]}\n";
    std::fwrite(line.data(), 1, line.size(), file);
    std::fflush(file);
}
