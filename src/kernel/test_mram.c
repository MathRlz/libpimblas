#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <perfcounter.h>

struct params {
  uint32_t no_tasklets;
  uint32_t num_elems_mram;
  uint32_t num_elems_wram_per_tasklet;
  int step;
};

__host struct params args;

__host perfcounter_pair_t seq_read_and_add_int_results[NR_TASKLETS];
__host perfcounter_pair_t seq_rw_and_add_int_results[NR_TASKLETS];
__host perfcounter_pair_t seq_read_and_mul_int_results[NR_TASKLETS];
__host perfcounter_pair_t seq_rw_and_mul_int_results[NR_TASKLETS];
__host perfcounter_pair_t seq_read_and_div_int_results[NR_TASKLETS];
__host perfcounter_pair_t seq_rw_and_div_int_results[NR_TASKLETS];

__host perfcounter_pair_t seq_read_and_add_float_results[NR_TASKLETS];
__host perfcounter_pair_t seq_rw_and_add_float_results[NR_TASKLETS];
__host perfcounter_pair_t seq_read_and_mul_float_results[NR_TASKLETS];
__host perfcounter_pair_t seq_rw_and_mul_float_results[NR_TASKLETS];
__host perfcounter_pair_t seq_read_and_div_float_results[NR_TASKLETS];
__host perfcounter_pair_t seq_rw_and_div_float_results[NR_TASKLETS];

inline void start_count() { perfcounter_config(COUNT_ENABLE_BOTH, true); }
inline void end_count(perfcounter_pair_t *counters) { counters[me()] = perfcounter_get_both(false); }

BARRIER_INIT(mem_reset_barrier, NR_TASKLETS);

void read_and_add_int(int *mram_ptr, int *wram_ptr, int num_elems, int block_size, int step) {
  unsigned int num_blocks = (num_elems - 1) / block_size + 1;

  for (unsigned int i = 0; i < num_blocks; i++) {
    unsigned int block_offset = i * block_size;
    unsigned int block_length = block_offset + block_size <= num_elems ? block_size : num_elems - block_offset;
    mram_read((__mram_ptr void *)(mram_ptr + block_offset), wram_ptr, block_length * sizeof(int));
    volatile int a = 0;
    for (unsigned int j = 0; j < block_length; j += step) {
      a += wram_ptr[j];
    }
  }
}

void rw_and_add_int(int *mram_ptr, int *wram_ptr, int num_elems, int block_size, int step) {
  unsigned int num_blocks = (num_elems - 1) / block_size + 1;

  for (unsigned int i = 0; i < num_blocks; i++) {
    unsigned int block_offset = i * block_size;
    unsigned int block_length = block_offset + block_size <= num_elems ? block_size : num_elems - block_offset;
    mram_read((__mram_ptr void *)(mram_ptr + block_offset), wram_ptr, block_length * sizeof(int));
    volatile int a = 0;
    for (unsigned int j = 0; j < block_length; j += step) {
      a += wram_ptr[j];
      wram_ptr[j] = a;
    }
    mram_write(wram_ptr, (__mram_ptr void *)(mram_ptr + block_offset), block_length * sizeof(int));
  }
}

void read_and_mul_int(int *mram_ptr, int *wram_ptr, int num_elems, int block_size, int step) {
  unsigned int num_blocks = (num_elems - 1) / block_size + 1;

  for (unsigned int i = 0; i < num_blocks; i++) {
    unsigned int block_offset = i * block_size;
    unsigned int block_length = block_offset + block_size <= num_elems ? block_size : num_elems - block_offset;
    mram_read((__mram_ptr void *)(mram_ptr + block_offset), wram_ptr, block_length * sizeof(int));
    volatile int a = 0;
    for (unsigned int j = 0; j < block_length; j += step) {
      a *= wram_ptr[j];
    }
  }
}

void rw_and_mul_int(int *mram_ptr, int *wram_ptr, int num_elems, int block_size, int step) {
  unsigned int num_blocks = (num_elems - 1) / block_size + 1;

  for (unsigned int i = 0; i < num_blocks; i++) {
    unsigned int block_offset = i * block_size;
    unsigned int block_length = block_offset + block_size <= num_elems ? block_size : num_elems - block_offset;
    mram_read((__mram_ptr void *)(mram_ptr + block_offset), wram_ptr, block_length * sizeof(int));
    volatile int a = 0;
    for (unsigned int j = 0; j < block_length; j += step) {
      a *= wram_ptr[j];
      wram_ptr[j] = a;
    }
    mram_write(wram_ptr, (__mram_ptr void *)(mram_ptr + block_offset), block_length * sizeof(int));
  }
}

void read_and_div_int(int *mram_ptr, int *wram_ptr, int num_elems, int block_size, int step) {
  unsigned int num_blocks = (num_elems - 1) / block_size + 1;

  for (unsigned int i = 0; i < num_blocks; i++) {
    unsigned int block_offset = i * block_size;
    unsigned int block_length = block_offset + block_size <= num_elems ? block_size : num_elems - block_offset;
    mram_read((__mram_ptr void *)(mram_ptr + block_offset), wram_ptr, block_length * sizeof(int));
    volatile int a = 0;
    for (unsigned int j = 0; j < block_length; j += step) {
      if (wram_ptr[j] != 0) {
        a /= wram_ptr[j];
      }
    }
  }
}

void rw_and_div_int(int *mram_ptr, int *wram_ptr, int num_elems, int block_size, int step) {
  unsigned int num_blocks = (num_elems - 1) / block_size + 1;

  for (unsigned int i = 0; i < num_blocks; i++) {
    unsigned int block_offset = i * block_size;
    unsigned int block_length = block_offset + block_size <= num_elems ? block_size : num_elems - block_offset;
    mram_read((__mram_ptr void *)(mram_ptr + block_offset), wram_ptr, block_length * sizeof(int));
    volatile int a = 0;
    for (unsigned int j = 0; j < block_length; j += step) {
      if (wram_ptr[j] != 0) {
        a /= wram_ptr[j];
      }
      wram_ptr[j] = a;
    }
    mram_write(wram_ptr, (__mram_ptr void *)(mram_ptr + block_offset), block_length * sizeof(int));
  }
}

void read_and_add_float(float *mram_ptr, float *wram_ptr, int num_elems, int block_size, int step) {
  unsigned int num_blocks = (num_elems - 1) / block_size + 1;

  for (unsigned int i = 0; i < num_blocks; i++) {
    unsigned int block_offset = i * block_size;
    unsigned int block_length = block_offset + block_size <= num_elems ? block_size : num_elems - block_offset;
    mram_read((__mram_ptr void *)(mram_ptr + block_offset), wram_ptr, block_length * sizeof(float));
    volatile float a = 0;
    for (unsigned int j = 0; j < block_length; j += step) {
      a += wram_ptr[j];
    }
  }
}

void rw_and_add_float(float *mram_ptr, float *wram_ptr, int num_elems, int block_size, int step) {
  unsigned int num_blocks = (num_elems - 1) / block_size + 1;

  for (unsigned int i = 0; i < num_blocks; i++) {
    unsigned int block_offset = i * block_size;
    unsigned int block_length = block_offset + block_size <= num_elems ? block_size : num_elems - block_offset;
    mram_read((__mram_ptr void *)(mram_ptr + block_offset), wram_ptr, block_length * sizeof(float));
    volatile float a = 0;
    for (unsigned int j = 0; j < block_length; j += step) {
      a += wram_ptr[j];
      wram_ptr[j] = a;
    }
    mram_write(wram_ptr, (__mram_ptr void *)(mram_ptr + block_offset), block_length * sizeof(float));
  }
}

void read_and_mul_float(float *mram_ptr, float *wram_ptr, int num_elems, int block_size, int step) {
  unsigned int num_blocks = (num_elems - 1) / block_size + 1;

  for (unsigned int i = 0; i < num_blocks; i++) {
    unsigned int block_offset = i * block_size;
    unsigned int block_length = block_offset + block_size <= num_elems ? block_size : num_elems - block_offset;
    mram_read((__mram_ptr void *)(mram_ptr + block_offset), wram_ptr, block_length * sizeof(float));
    volatile float a = 0;
    for (unsigned int j = 0; j < block_length; j += step) {
      a *= wram_ptr[j];
    }
  }
}

void rw_and_mul_float(float *mram_ptr, float *wram_ptr, int num_elems, int block_size, int step) {
  unsigned int num_blocks = (num_elems - 1) / block_size + 1;

  for (unsigned int i = 0; i < num_blocks; i++) {
    unsigned int block_offset = i * block_size;
    unsigned int block_length = block_offset + block_size <= num_elems ? block_size : num_elems - block_offset;
    mram_read((__mram_ptr void *)(mram_ptr + block_offset), wram_ptr, block_length * sizeof(float));
    volatile float a = 0;
    for (unsigned int j = 0; j < block_length; j += step) {
      a *= wram_ptr[j];
      wram_ptr[j] = a;
    }
    mram_write(wram_ptr, (__mram_ptr void *)(mram_ptr + block_offset), block_length * sizeof(float));
  }
}

void read_and_div_float(float *mram_ptr, float *wram_ptr, int num_elems, int block_size, int step) {
  unsigned int num_blocks = (num_elems - 1) / block_size + 1;

  for (unsigned int i = 0; i < num_blocks; i++) {
    unsigned int block_offset = i * block_size;
    unsigned int block_length = block_offset + block_size <= num_elems ? block_size : num_elems - block_offset;
    mram_read((__mram_ptr void *)(mram_ptr + block_offset), wram_ptr, block_length * sizeof(float));
    volatile float a = 0;
    for (unsigned int j = 0; j < block_length; j += step) {
      if (wram_ptr[j] != 0) {
        a /= wram_ptr[j];
      }
    }
  }
}

void rw_and_div_float(float *mram_ptr, float *wram_ptr, int num_elems, int block_size, int step) {
  unsigned int num_blocks = (num_elems - 1) / block_size + 1;

  for (unsigned int i = 0; i < num_blocks; i++) {
    unsigned int block_offset = i * block_size;
    unsigned int block_length = block_offset + block_size <= num_elems ? block_size : num_elems - block_offset;
    mram_read((__mram_ptr void *)(mram_ptr + block_offset), wram_ptr, block_length * sizeof(float));
    volatile float a = 0;
    for (unsigned int j = 0; j < block_length; j += step) {
      if (wram_ptr[j] != 0) {
        a /= wram_ptr[j];
      }
      wram_ptr[j] = a;
    }
    mram_write(wram_ptr, (__mram_ptr void *)(mram_ptr + block_offset), block_length * sizeof(float));
  }
}

int main() {
  int tasklet_id = me();
  if (tasklet_id == 0) {
    mem_reset();
  }
  barrier_wait(&mem_reset_barrier);

  if (me() >= args.no_tasklets) {
    return 0;
  }

  void *my_wram = mem_alloc(sizeof(int) * args.num_elems_wram_per_tasklet);

  unsigned int elems_per_tasklet = args.num_elems_mram / args.no_tasklets;
  start_count();
  read_and_add_int((int *)DPU_MRAM_HEAP_POINTER + me() * elems_per_tasklet, (int *)my_wram, elems_per_tasklet,
                   args.num_elems_wram_per_tasklet, args.step);
  end_count(seq_read_and_add_int_results);

  start_count();
  rw_and_add_int((int *)DPU_MRAM_HEAP_POINTER + me() * elems_per_tasklet, (int *)my_wram, elems_per_tasklet,
                 args.num_elems_wram_per_tasklet, args.step);
  end_count(seq_rw_and_add_int_results);

  start_count();
  read_and_mul_int((int *)DPU_MRAM_HEAP_POINTER + me() * elems_per_tasklet, (int *)my_wram, elems_per_tasklet,
                   args.num_elems_wram_per_tasklet, args.step);
  end_count(seq_read_and_mul_int_results);

  start_count();
  rw_and_mul_int((int *)DPU_MRAM_HEAP_POINTER + me() * elems_per_tasklet, (int *)my_wram, elems_per_tasklet,
                 args.num_elems_wram_per_tasklet, args.step);
  end_count(seq_rw_and_mul_int_results);

  start_count();
  read_and_div_int((int *)DPU_MRAM_HEAP_POINTER + me() * elems_per_tasklet, (int *)my_wram, elems_per_tasklet,
                   args.num_elems_wram_per_tasklet, args.step);
  end_count(seq_read_and_div_int_results);

  start_count();
  rw_and_div_int((int *)DPU_MRAM_HEAP_POINTER + me() * elems_per_tasklet, (int *)my_wram, elems_per_tasklet,
                 args.num_elems_wram_per_tasklet, args.step);
  end_count(seq_rw_and_div_int_results);

  start_count();
  read_and_add_float((float *)DPU_MRAM_HEAP_POINTER + me() * elems_per_tasklet, (float *)my_wram, elems_per_tasklet,
                     args.num_elems_wram_per_tasklet, args.step);
  end_count(seq_read_and_add_float_results);

  start_count();
  rw_and_add_float((float *)DPU_MRAM_HEAP_POINTER + me() * elems_per_tasklet, (float *)my_wram, elems_per_tasklet,
                   args.num_elems_wram_per_tasklet, args.step);
  end_count(seq_rw_and_add_float_results);

  start_count();
  read_and_mul_float((float *)DPU_MRAM_HEAP_POINTER + me() * elems_per_tasklet, (float *)my_wram, elems_per_tasklet,
                     args.num_elems_wram_per_tasklet, args.step);
  end_count(seq_read_and_mul_float_results);

  start_count();
  rw_and_mul_float((float *)DPU_MRAM_HEAP_POINTER + me() * elems_per_tasklet, (float *)my_wram, elems_per_tasklet,
                   args.num_elems_wram_per_tasklet, args.step);
  end_count(seq_rw_and_mul_float_results);

  start_count();
  read_and_div_float((float *)DPU_MRAM_HEAP_POINTER + me() * elems_per_tasklet, (float *)my_wram, elems_per_tasklet,
                     args.num_elems_wram_per_tasklet, args.step);
  end_count(seq_read_and_div_float_results);

  start_count();
  rw_and_div_float((float *)DPU_MRAM_HEAP_POINTER + me() * elems_per_tasklet, (float *)my_wram, elems_per_tasklet,
                   args.num_elems_wram_per_tasklet, args.step);
  end_count(seq_rw_and_div_float_results);

  return 0;
}