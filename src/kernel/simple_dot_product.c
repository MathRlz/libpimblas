#include <defs.h>
#include <mram.h>
#include <stdio.h>
#include <barrier.h>

__host uint64_t offset;
__host uint64_t result;
__host uint64_t num_elems;

#define BLOCK_SIZE 256

#define PERFCOUNT on

#ifdef PERFCOUNT
#include <perfcounter.h>
__host uint32_t nb_cycles[NR_TASKLETS];
__host uint32_t nb_instructions[NR_TASKLETS];
BARRIER_INIT(perfcount_start_barrier, NR_TASKLETS);
#endif

__dma_aligned uint32_t vec1[NR_TASKLETS][BLOCK_SIZE];
__dma_aligned uint32_t vec2[NR_TASKLETS][BLOCK_SIZE];
__dma_aligned uint64_t results[NR_TASKLETS];

BARRIER_INIT(gather_results_barrier, NR_TASKLETS);

uint64_t dot_product(uint32_t *vec1, uint32_t *vec2, uint32_t size) {
    uint64_t result = 0;
    for (uint32_t i = 0; i < size; i++) {
        result += (uint64_t)vec1[i] * (uint64_t)vec2[i];
    }
    return result;
}

int main() {
    #ifdef PERFCOUNT
    if (me() == 0) {
        perfcounter_config(COUNT_ENABLE_BOTH, true);
    }
    barrier_wait(&perfcount_start_barrier);
    #endif

    int tasklet_id = me();
    int number_blocks = (num_elems - 1) / BLOCK_SIZE + 1;
    int number_blocks_per_tasklet = (number_blocks - 1) / NR_TASKLETS + 1;
    int tasklet_block_start = tasklet_id * number_blocks_per_tasklet;
    int blocks_per_tasklet_size = number_blocks_per_tasklet * BLOCK_SIZE * sizeof(uint32_t);

    uint32_t *vec1_MRAM = (uint32_t*)(DPU_MRAM_HEAP_POINTER
                           +  tasklet_id * blocks_per_tasklet_size);
    uint32_t *vec2_MRAM = (uint32_t*)(DPU_MRAM_HEAP_POINTER
                           + offset + tasklet_id * blocks_per_tasklet_size);

    results[tasklet_id] = 0;
    for (uint32_t block_id = 0; block_id < number_blocks_per_tasklet; block_id++) {
        int block_offset = block_id * BLOCK_SIZE;

        int32_t block_length = BLOCK_SIZE;
        if (tasklet_block_start * BLOCK_SIZE + BLOCK_SIZE >= num_elems) {
            block_length = num_elems - tasklet_block_start * BLOCK_SIZE;
            if (block_length <= 0) {
                break;
            }
        } 

        mram_read((__mram_ptr void*)(vec1_MRAM + block_offset), vec1[tasklet_id], block_length * sizeof(uint32_t));
        mram_read((__mram_ptr void*)(vec2_MRAM + block_offset), vec2[tasklet_id], block_length * sizeof(uint32_t));
        
        results[tasklet_id] += dot_product(vec1[tasklet_id], vec2[tasklet_id], block_length);
    }

    barrier_wait(&gather_results_barrier);
    if (tasklet_id == 0) {
        result = 0;
        for (int i = 0; i < NR_TASKLETS; i++) {
            result += results[i];
        }
    }

    #ifdef PERFCOUNT
    perfcounter_pair_t counters = perfcounter_get_both(false);
    nb_cycles[me()] = counters.cycles;
    nb_instructions[me()] = counters.instr;
    #endif

    return 0;
}