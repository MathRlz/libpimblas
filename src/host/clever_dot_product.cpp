#include "common.hpp"
#include "kernel.hpp"

template <typename T>
T alignUp(T value, size_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

template <typename T>
std::vector<uint64_t> transposeBits(T *src, size_t size) {
    constexpr size_t bitsPerByte = 8;
    constexpr size_t bitsPerLong = sizeof(uint64_t) * bitsPerByte;
    constexpr size_t bitsPerElement = sizeof(T) * bitsPerByte;

    size_t totalBits = size * bitsPerElement;
    size_t destSize = alignUp((totalBits + bitsPerLong - 1) / bitsPerLong, 32);

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

    size_t trSize = tr1.size();
    // trChunkSize needs to be aligned to 32 
    //size_t trChunkSize = 2 * 1024 * 1024; // Max size
    size_t trChunkSize = 1024 * 8; // Optimal size for speed somewhere between 2K and 8K
    if (trChunkSize > trSize) {
        trChunkSize = trSize;
    }

    size_t remainder = trSize % trChunkSize;
    uint32_t nr_dpus = (trSize - 1) / trChunkSize + 1;

    Kernel kernel;
    if (false == kernel.allocate_dpus(nr_dpus)) {
        show_error("clever_dot_product: couldn't allocate required ({}) nr dpus.", nr_dpus);
        return -1;
    }
    kernel.load_program("clever_dot_product.kernel");

    kernel.set_arg_broadcast_exact("num_elems", 0, &trChunkSize, sizeof(size_t), false);
    if (remainder != 0) {
        dpu_set_t dpu;
        uint32_t dpu_idx;
        DPU_FOREACH(kernel.get_dpu_set(), dpu, dpu_idx) {
            if (dpu_idx == nr_dpus - 1) {
                dpu_copy_to(dpu, "num_elems", 0, &remainder, sizeof(size_t));
            }
        }
    }

    constexpr size_t precision = 32;
    size_t number_blocks = (trChunkSize) / precision;
    size_t offset = number_blocks * precision * sizeof(uint64_t);
    kernel.set_arg_broadcast_exact("offset", 0, &offset, sizeof(size_t), false);

    kernel.set_arg_scatter(DPU_MRAM_HEAP_POINTER_NAME, 0, tr1.data(), trChunkSize * sizeof(uint64_t), trSize * sizeof(uint64_t), false);
    kernel.set_arg_scatter(DPU_MRAM_HEAP_POINTER_NAME, offset, tr2.data(), trChunkSize * sizeof(uint64_t), trSize * sizeof(uint64_t), false);

    kernel.launch(false);

    std::vector<uint64_t> results(nr_dpus);
    kernel.get_arg_copy_each("result", 0, results.data(), sizeof(uint64_t));

    *result = 0U;
    for (auto &partial_result : results) {
        *result += partial_result;
    }

    kernel.free_dpus();
    return 0;
}
}