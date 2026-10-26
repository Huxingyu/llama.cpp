#pragma once

#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

// Optional machine-readable evidence. Token IDs are recorded before EOG
// postprocessing; discarded verification suffixes appear only in verify events.
inline void common_mtp_trace(const char * event, int round, const std::vector<llama_token> & tokens = {},
                             int drafted = 0, int accepted = 0, int64_t elapsed_us = 0, bool restore = false) {
    static const char * path = std::getenv("LLAMA_MTP_TRACE");
    if (!path || !*path) { return; }
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    static FILE * file = [] {
        FILE * result = std::fopen(path, "a");
        if (!result) { std::perror("LLAMA_MTP_TRACE"); std::abort(); }
        return result;
    }();
    std::fprintf(file, "{\"event\":\"%s\",\"round\":%d,\"drafted\":%d,\"accepted\":%d,\"elapsed_us\":%lld,\"restore\":%s,\"tokens\":[",
                 event, round, drafted, accepted, (long long) elapsed_us, restore ? "true" : "false");
    for (size_t i = 0; i < tokens.size(); ++i) { std::fprintf(file, "%s%d", i ? "," : "", tokens[i]); }
    std::fprintf(file, "]}\n");
    std::fflush(file);
}
