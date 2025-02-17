#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>

__host uint64_t offset;
__host uint64_t result;
__host uint64_t num_elems;

unsigned int count_all_ones_32(uint32_t value) {
    unsigned int result;
    __asm__ volatile ("cao %0, %1" : "=r"(result) : "r"(value));
    return result;
}

unsigned int count_all_ones_64(uint64_t value) {
    uint32_t lower = (uint32_t)(value & 0xFFFFFFFF);
    uint32_t higher = (uint32_t)(value >> 32);

    unsigned int lower_ones = count_all_ones_32(lower);
    unsigned int higher_ones = count_all_ones_32(higher);

    return lower_ones + higher_ones;
}

#define PRECISION 32
#define BLOCK_SIZE 5

uint64_t clever_dot_product(const uint64_t *arr1, const uint64_t *arr2, uint32_t size) {
    uint64_t dp = 0;

    for (int i = 0; i < size; i += PRECISION) {
        int exp = (PRECISION - 1) << 1;

        for (; exp >= PRECISION; --exp) {
            uint64_t partSum = 0;

            for (int j = exp - PRECISION + 1; j <= PRECISION - 1; ++j) {
                partSum += count_all_ones_64(arr1[i+j] & arr2[i+exp-j]);
            }

            dp += partSum << exp;
        }

        for (; exp >= 0; --exp) {
            uint64_t partSum = 0;

            for (int j = 0; j <= exp; ++j) {
                partSum += count_all_ones_64(arr1[i+j] & arr2[i+exp-j]);
            }
            
            dp += partSum << exp;
        }
    }

    return dp;
}


__dma_aligned uint64_t vec1[NR_TASKLETS][PRECISION];
__dma_aligned uint64_t vec2[NR_TASKLETS][PRECISION];
__dma_aligned uint64_t tmpResults[NR_TASKLETS];

BARRIER_INIT(gather_results_barrier, NR_TASKLETS);

int main() {
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

  for (uint32_t block_id = 0; block_id < number_blocks_per_tasklet; block_id++) {
    const int block_offset = block_id * PRECISION;

    if (tasklet_block_start + block_id >= number_blocks) {
      break;
    }

    mram_read((__mram_ptr void *)(vec1_MRAM + block_offset), vec1[tasklet_id], PRECISION * sizeof(uint64_t));
    mram_read((__mram_ptr void *)(vec2_MRAM + block_offset), vec2[tasklet_id], PRECISION * sizeof(uint64_t));

    tmpResults[tasklet_id] += clever_dot_product(vec1[tasklet_id], vec2[tasklet_id], PRECISION);
  }

  barrier_wait(&gather_results_barrier);
  if (tasklet_id == 0) {
    result = 0;
    for (int i = 0; i < NR_TASKLETS; i++) {
        result += tmpResults[i];
    }
  }

  return 0;
}