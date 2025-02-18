#include "perfcount_helper.hpp"
#include "common.hpp"
#include "kernel.hpp"

extern "C" {
int dot_product(const uint32_t *vec1, const uint32_t *vec2, size_t vecSize, uint64_t *result) {
    
    size_t chunkSize = 8 * 1024 * 1024;
    if (chunkSize > vecSize) {
        chunkSize = vecSize;
    }

    size_t remainder = vecSize % chunkSize;
    uint32_t nr_dpus = (vecSize - 1) / chunkSize + 1;

    Kernel kernel;
    if (false == kernel.allocate_dpus(nr_dpus)) {
        show_error("dot_product: couldn't allocate required ({}) nr dpus.", nr_dpus);
        return -1;
    }
    kernel.load_program("simple_dot_product.kernel");

    kernel.set_arg_broadcast_exact("num_elems", 0, &chunkSize, sizeof(size_t), false);
    if (remainder != 0) {
        dpu_set_t dpu;
        uint32_t dpu_idx;
        DPU_FOREACH(kernel.get_dpu_set(), dpu, dpu_idx) {
            if (dpu_idx == nr_dpus - 1) {
                dpu_copy_to(dpu, "num_elems", 0, &remainder, sizeof(size_t));
            }
        }
    }

    size_t offset = chunkSize * sizeof(uint32_t);
    kernel.set_arg_broadcast_exact("offset", 0, &offset, sizeof(size_t), false);

    kernel.set_arg_scatter(DPU_MRAM_HEAP_POINTER_NAME, 0, vec1, chunkSize * sizeof(uint32_t), vecSize * sizeof(uint32_t), false);
    kernel.set_arg_scatter(DPU_MRAM_HEAP_POINTER_NAME, offset, vec2, chunkSize * sizeof(uint32_t), vecSize * sizeof(uint32_t), false);

    kernel.launch(false);

    std::vector<uint64_t> results(nr_dpus);
    kernel.get_arg_copy_each("result", 0, results.data(), sizeof(uint64_t));

    *result = 0U;
    for (auto &partial_result : results) {
        *result += partial_result;
    }

    constexpr bool gatherPfResults = true;
    if constexpr (gatherPfResults) {
        auto perfRes = get_pf_results(kernel.get_dpu_set());
        for (auto &perf : perfRes) {
            std::cout << "cycles: " << perf.nb_cycles 
                    << ", instr: " << perf.nb_instr << std::endl;
        }
    }

    kernel.free_dpus();
    return 0;
}
}