// Public backend buffer contract: canonical bytes must survive physical repacking.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

static bool check(ggml_backend_dev_t dev, ggml_type type, int64_t width, int64_t rows, int64_t experts) {
    ggml_context * ctx = ggml_init({ggml_tensor_overhead() * 2, nullptr, true});
    if (!ctx) { return false; }
    ggml_tensor * tensor = ggml_new_tensor_3d(ctx, type, width, rows, experts);
    ggml_tensor * cpu = ggml_dup_tensor(ctx, tensor);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_dev_buffer_type(dev));
    if (!buffer) { ggml_free(ctx); return false; }
    // Use a separate CPU tensor/context for cross-backend copies.
    ggml_context * host = ggml_init({ggml_tensor_overhead(), nullptr, true});
    ggml_tensor * destination = ggml_dup_tensor(host, cpu);
    ggml_backend_buffer_t host_buffer = ggml_backend_alloc_ctx_tensors_from_buft(host, ggml_backend_cpu_buffer_type());
    const size_t size = ggml_nbytes(tensor);
    std::vector<uint8_t> expected(size), actual(size);
    std::vector<float> input(ggml_nelements(tensor));
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = float(int((i * 97 + 13) % 211) - 105) / 31.0f;
    }
    ggml_quantize_init(type);
    bool ok = ggml_quantize_chunk(type, input.data(), expected.data(), 0, rows * experts, width, nullptr) == size;
    ggml_backend_tensor_set(tensor, expected.data(), 0, size);
    ggml_backend_tensor_get(tensor, actual.data(), 0, size);
    ok = ok && actual == expected;
    for (size_t offset : {size_t(1), size_t(17), size / 2 - 3, size - 19}) {
        std::vector<uint8_t> slice(19);
        ggml_backend_tensor_get(tensor, slice.data(), offset, slice.size());
        ok = ok && std::equal(slice.begin(), slice.end(), expected.begin() + offset);
    }
    // An unaligned partial upload spans scale and quant bytes. Values need not
    // be meaningful floats: the buffer API promises exact byte preservation.
    const uint8_t patch[] = {0x3c, 0x01, 0x87, 0x9f, 0x42};
    std::copy(std::begin(patch), std::end(patch), expected.begin() + 17);
    ggml_backend_tensor_set(tensor, patch, 17, sizeof(patch));
    ggml_backend_tensor_copy(tensor, destination);
    ggml_backend_tensor_get(destination, actual.data(), 0, size);
    ok = ok && actual == expected;
    ggml_backend_tensor_copy(destination, cpu);
    ggml_backend_tensor_get(cpu, actual.data(), 0, size);
    ok = ok && actual == expected;
    ggml_backend_tensor_copy(tensor, cpu);
    ggml_backend_tensor_get(cpu, actual.data(), 0, size);
    ok = ok && actual == expected;
    ggml_backend_tensor_memset(tensor, 0, 13, 23);
    std::fill(expected.begin() + 13, expected.begin() + 36, 0);
    ggml_backend_tensor_get(tensor, actual.data(), 0, size);
    ok = ok && actual == expected;
    printf("BUFFER type=%s width=%lld rows=%lld experts=%lld bytes=%zu result=%s\n", ggml_type_name(type),
           (long long) width, (long long) rows, (long long) experts, size, ok ? "PASS" : "FAIL");
    ggml_backend_buffer_free(host_buffer);
    ggml_free(host);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("SPACEMIT0");
    if (!dev) { fprintf(stderr, "SPACEMIT0 is required; no cases executed\n"); return 1; }
    size_t passed = 0, total = 0;
    for (ggml_type type : {GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q4_K, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0}) {
        for (int64_t rows : {16, 32, 48}) {
            for (int64_t experts : {1, 2}) {
                ++total;
                passed += check(dev, type, 256, rows, experts);
            }
        }
    }
    printf("BUFFER_COUNTS executed=%zu passed=%zu failed=%zu\n", total, passed, total - passed);
    return total > 0 && passed == total ? 0 : 1;
}
