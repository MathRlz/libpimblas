#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <stdint.h>
#include <string.h>

/*
Basic GEMV kernel performing A * x
A is a matrix of size m x n,
x is a vector of size n

Notes:
Only part of A is transferred to single DPU. Namely rows_per_dpu
x is same across all DPU's

Computing parameters:
NR_TASKLETS - number of tasklets (threads) running on single DPU
rows_per_dpu - maximum number of rows to be processed by single DPU
row_size - maximum size of single matrix row

*/

// We've got 64KB of WRAM, we are working with 4B floats, and need to allocate wram
// for part of A rows, and part of X rows and output(small in comparison), and 16 tasklets.
// That makes 4KB per tasklet, that means we could go to block size 512 - aka 4KB, but that
// would leave no place for output. So we stick with 256 for now.
// TODO: Find even more optimal value
#define BLOCK_SIZE 256

struct params {
  uint32_t rows_per_dpu;
  uint32_t row_size;
  float alpha;
};

__host struct params args;

BARRIER_INIT(mem_reset_barrier, NR_TASKLETS);

uint32_t alignUpTo8(uint32_t value) { return (value + 7) & ~7; }

uint32_t alignDownTo8(uint32_t value) { return value & ~7; }

uint32_t alignUpTo64(uint32_t value) { return (value + 63) & ~63; }

uint32_t alignUpTo2(uint32_t value) { return (value + 1) & ~1; }

int main() {
  int tasklet_id = me();
  if (tasklet_id == 0) {
    mem_reset();
  }
  barrier_wait(&mem_reset_barrier);

  // Sanity checks: NR_tasklets should be 16, rows_per_dpu should be a multiple of 32, because
  // rows per tasklet should be even
  if (NR_TASKLETS != 16 || args.rows_per_dpu & 31) {
    return 1;
  }
  // Rows per tasklet
  int rows_per_tasklet = args.rows_per_dpu / NR_TASKLETS;

  // Note: All MRAM allocations need to be 8B aligned in order to read from/write to them.

  // Offset of A_mram should be 8B aligned,
  // even if row_size is odd, rows_per_tasklet is always aligned to 2,
  // so it should be fine, because we are operating 4B floats.
  uint32_t mram_offset_in_bytes = 0;

  float *A_mram = (float *)(DPU_MRAM_HEAP_POINTER + mram_offset_in_bytes +
                            (tasklet_id * args.row_size * rows_per_tasklet) * sizeof(float));
  mram_offset_in_bytes += alignUpTo8(args.row_size * args.rows_per_dpu * sizeof(float));

  float *x_mram = (float *)(DPU_MRAM_HEAP_POINTER + mram_offset_in_bytes);
  mram_offset_in_bytes += alignUpTo8(args.row_size * sizeof(float));

  // Should be fine as long as rows_per_tasklet is even
  float *result_mram =
      (float *)(DPU_MRAM_HEAP_POINTER + mram_offset_in_bytes + (tasklet_id * rows_per_tasklet) * sizeof(float));

  // TODO: Find better way to share x across all tasklets, because now we
  // have multiple copies of the same values across tasklets.
  // If number of rows to be processed is small enough it should be possible
  // or we could just make a barrier and wait until all tasklets finish until
  // getting another part of x
  float *x_wram = (float *)mem_alloc(BLOCK_SIZE * sizeof(float));
  // It's important we allocate more memory for A_wram, because of the hack
  // we later to do to write into it from mram (alignment issues).
  // We add 64B in order to be aligned.
  float *A_wram = (float *)mem_alloc((BLOCK_SIZE) * sizeof(float) + 64);

  // Allocation needs to be aligned to 64B, or we start getting
  // allocations on top of another...
  uint32_t result_size = alignUpTo64(rows_per_tasklet * sizeof(float));
  float *result_wram = (float *)mem_alloc(result_size);

  // zero out the results - it's required when we are running the kernel multiple times.
  memset(result_wram, 0, result_size);

  int nr_blocks = (args.row_size - 1) / BLOCK_SIZE + 1;
  for (int block = 0; block < nr_blocks; block++) {
    const int block_offset = block * BLOCK_SIZE;

    int block_length = block_offset + BLOCK_SIZE <= args.row_size ? BLOCK_SIZE : args.row_size - block_offset;
    mram_read((__mram_ptr void *)(x_mram + block_offset), x_wram, BLOCK_SIZE * sizeof(float));
    for (int i = 0; i < rows_per_tasklet; i++) {
      float sum = 0;
      uint32_t a_offset = (uint32_t)(A_mram + i * args.row_size + block_offset);
      float *A_wram_read = NULL;
      if (a_offset & 7) {
        // If offset is not aligned to 8B it will be automatically aligned down to 8 bytes
        // This happens when row_size is an odd value.
        // In our case when we are working on 4B floats it means we need to shift
        // one float (4B) to get to the values we want. That also means we need to read a bit more
        mram_read((__mram_ptr void *)(alignDownTo8(a_offset)), A_wram, (BLOCK_SIZE + 2) * sizeof(float));
        A_wram_read = (A_wram + 1);
      } else {
        mram_read((__mram_ptr void *)(a_offset), A_wram, BLOCK_SIZE * sizeof(float));
        A_wram_read = A_wram;
      }

      for (int j = 0; j < block_length; ++j) {
        sum += A_wram_read[j] * x_wram[j];
      }

      result_wram[i] += sum;
    }
  }

  if (args.alpha != 1.0f) {
    for (uint32_t i = 0; i < rows_per_tasklet; i++) {
      result_wram[i] *= args.alpha;
    }
  }

  mram_write(result_wram, (__mram_ptr void *)result_mram, rows_per_tasklet * sizeof(float));

  return 0;
}