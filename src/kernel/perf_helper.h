#include <barrier.h>
#include <defs.h>
#include <perfcounter.h>

__host uint32_t nb_cycles[NR_TASKLETS];
__host uint32_t nb_instructions[NR_TASKLETS];
BARRIER_INIT(perfcount_start_barrier, NR_TASKLETS);

void perfcount_start() {
  if (me() == 0) {
    perfcounter_config(COUNT_ENABLE_BOTH, true);
  }
  barrier_wait(&perfcount_start_barrier);
}

void perfcount_stop() {
  perfcounter_pair_t counters = perfcounter_get_both(false);
  nb_cycles[me()] = counters.cycles;
  nb_instructions[me()] = counters.instr;
}