#include "perfcount_helper.hpp"
#include <array>

std::vector<PerfResults> get_pf_results(dpu_set_t dpu_set) {
 std::vector<PerfResults> results;

 dpu_set_t dpu;
 DPU_FOREACH(dpu_set, dpu) {
    std::array<uint32_t, 16> nb_cycles;
    std::array<uint32_t, 16> nb_instr;
    DPU_ASSERT(dpu_copy_from(dpu, "nb_cycles", 0, nb_cycles.data(), sizeof(uint32_t) * 16));
    DPU_ASSERT(dpu_copy_from(dpu, "nb_instructions", 0, nb_instr.data(), sizeof(uint32_t) * 16));

    results.push_back(PerfResults {
        .nb_cycles = *std::max_element(nb_cycles.begin(), nb_cycles.end()),
        .nb_instr = *std::max_element(nb_instr.begin(), nb_instr.end())
    });
 }

 return results;
}