#include "common.hpp"
#include "kernel.hpp"

template <typename T>
std::vector<uint64_t> transposeBits(T *src, size_t size) {
    constexpr size_t bitsPerByte = 8;
    constexpr size_t bitsPerLong = sizeof(uint64_t) * bitsPerByte;
    constexpr size_t bitsPerElement = sizeof(T) * bitsPerByte;

    size_t totalBits = size * bitsPerElement;
    size_t destSize = (totalBits + bitsPerLong - 1) / bitsPerLong;

    std::vector<uint64_t> dest(destSize, 0);

    for (size_t i = 0; i < size; i++) {
        size_t arrayOffset = (i / bitsPerLong) * bitsPerElement;
        size_t offset = (i / bitsPerElement * bitsPerElement) % bitsPerLong;

        for (size_t j = 0; j < bitsPerElement; ++j) {
            dest[arrayOffset + j] |= ((static_cast<uint64_t>(src[i]) >> j) & 1) << ((i % bitsPerElement) + offset);
        }
    }
    
    return dest;
}

extern "C" {
int clever_dot_product(const uint32_t *vec1, const uint32_t *vec2, size_t vecSize, uint64_t *result) {
    auto tr1 = transposeBits(vec1, vecSize);
    auto tr2 = transposeBits(vec2, vecSize);

    uint32_t nr_dpus = 1;
    dpu_set_t dpu_set;
    DPU_ASSERT(dpu_alloc(nr_dpus, nullptr, &dpu_set));

    Kernel kernel;
    kernel.set_dpu_set(dpu_set, nr_dpus);
    kernel.load_program("clever_dot_product.kernel");

    size_t trSize = tr1.size();
    //std::cout << "trSize: " <<trSize << std::endl;
    size_t trChunkSize = trSize;
    kernel.set_arg_broadcast_exact("vecSize", 0, &trChunkSize, sizeof(size_t), false);

    kernel.set_arg_scatter(DPU_MRAM_HEAP_POINTER_NAME, 0, tr1.data(), trChunkSize * sizeof(uint64_t), trSize * sizeof(uint64_t), false);

    size_t precision = 32;
    size_t number_blocks = (trChunkSize) / precision;
    size_t offset = number_blocks * precision * sizeof(uint64_t);
    //std::cout << "offset " << offset << std::endl;
    kernel.set_arg_scatter(DPU_MRAM_HEAP_POINTER_NAME, offset, tr2.data(), trChunkSize * sizeof(uint64_t), trSize * sizeof(uint64_t), false);

    kernel.launch(false);
    std::vector<uint64_t> results(nr_dpus);
    kernel.get_arg_copy_each("result", 0, results.data(), sizeof(uint64_t));
    //kernel.read_log();

    *result = 0U;
    for (auto &partial_result : results) {
        *result += partial_result;
    }

    kernel.free_dpus();
    return 0;
}
}