#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <stdint.h>
#include <string.h>


__host uint64_t offset;
__host uint64_t result;
__host uint64_t num_elems;

#define PERFCOUNT on

#ifdef PERFCOUNT
#include <perfcounter.h>
__host uint32_t nb_cycles[NR_TASKLETS];
__host uint32_t nb_instructions[NR_TASKLETS];
BARRIER_INIT(perfcount_start_barrier, NR_TASKLETS);
#endif

extern uint32_t count_all_ones_32(uint32_t value);
extern uint32_t count_all_ones_64(uint64_t value);
extern uint64_t fast_and(uint64_t valueA, uint64_t valueB);
extern uint64_t fast_loop(const uint64_t *arr1, const uint64_t *arr2);
extern uint64_t bit_shift(uint64_t value, int exponent);

#define PRECISION 32
#define BLOCK_SIZE 5

uint64_t clever_dot_product(const uint64_t *arr1, const uint64_t *arr2) {
    uint64_t dp = 0;

    int exp = (PRECISION - 1) << 1;

    for (; exp >= PRECISION; --exp) {
        uint64_t partSum = 0;

        for (int j = exp - PRECISION + 1; j <= PRECISION - 1; ++j) {
            uint64_t val1 = arr1[j];
            uint64_t val2 = arr2[exp-j];
            uint32_t val = count_all_ones_64(fast_and(val1, val2));
            partSum += val;
        }
        partSum = partSum << exp;
        dp += partSum;
    }

    for (; exp >= 0; --exp) {
        uint64_t partSum = 0;

        for (int j = 0; j <= exp; ++j) {
            partSum += count_all_ones_64(arr1[j] & arr2[exp-j]);
        }
        
        dp += partSum << exp;
    }

    return dp;
}

__dma_aligned uint64_t vec1[NR_TASKLETS][PRECISION * BLOCK_SIZE];
__dma_aligned uint64_t vec2[NR_TASKLETS][PRECISION * BLOCK_SIZE];
__dma_aligned uint64_t tmpResults[NR_TASKLETS];

BARRIER_INIT(gather_results_barrier, NR_TASKLETS);

int main() {
//  __dma_aligned volatile uint64_t wait = wait_value;
//  while (wait); // loops forever
  #ifdef PERFCOUNT
  if (me() == 0) {
      perfcounter_config(COUNT_ENABLE_BOTH, true);
  }
  barrier_wait(&perfcount_start_barrier);
  #endif
  int tasklet_id = me();

  int number_blocks = num_elems / PRECISION;
  int number_blocks_per_tasklet = (number_blocks - 1) / NR_TASKLETS + 1;
  int tasklet_block_start = tasklet_id * number_blocks_per_tasklet;
  int blocks_per_tasklet_size = number_blocks_per_tasklet * PRECISION * sizeof(uint64_t);

  uint64_t *vec1_MRAM = (uint64_t*)(DPU_MRAM_HEAP_POINTER
                         + tasklet_id * blocks_per_tasklet_size);
  uint64_t *vec2_MRAM = (uint64_t*)(DPU_MRAM_HEAP_POINTER
                         + offset // Offset to second vec
                         + tasklet_id * blocks_per_tasklet_size);
  
  tmpResults[tasklet_id] = 0;

  for (uint32_t block_id = 0; block_id < number_blocks_per_tasklet; block_id+=BLOCK_SIZE) {
    const int block_offset = block_id * PRECISION;

    const int num_blocks_left = number_blocks - tasklet_block_start - block_id;
    if (num_blocks_left <= 0) break;

    int num_blocks = BLOCK_SIZE;
    if (num_blocks > num_blocks_left) {
        num_blocks = num_blocks_left;
    }
    if (num_blocks > number_blocks_per_tasklet - block_id) {
        num_blocks = number_blocks_per_tasklet - block_id;
    }


    mram_read((__mram_ptr void *)(vec1_MRAM + block_offset), vec1[tasklet_id], num_blocks * PRECISION * sizeof(uint64_t));
    mram_read((__mram_ptr void *)(vec2_MRAM + block_offset), vec2[tasklet_id], num_blocks * PRECISION * sizeof(uint64_t));

    for (int i = 0; i < num_blocks; i++) {
        tmpResults[tasklet_id] += fast_loop(vec1[tasklet_id] + PRECISION * i, vec2[tasklet_id] + PRECISION * i);
    }
  }

  barrier_wait(&gather_results_barrier);
  if (tasklet_id == 0) {
    result = 0;
    for (int i = 0; i < NR_TASKLETS; i++) {
        result += tmpResults[i];
    }
  }

  #ifdef PERFCOUNT
  perfcounter_pair_t counters = perfcounter_get_both(false);
  nb_cycles[me()] = counters.cycles;
  nb_instructions[me()] = counters.instr;
  #endif

  return 0;
}