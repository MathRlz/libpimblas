#pragma once
#include "common.hpp"

struct PerfResults {
    uint32_t nb_cycles;
    uint32_t nb_instr;
};
std::vector<PerfResults> get_pf_results(dpu_set_t dpu_set);