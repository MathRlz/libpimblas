#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <stdio.h>

#define FLT_MIN 1.175494351e-38F

#define LOCAL_BUFFER_SIZE 512
__dma_aligned float vec_local[NR_TASKLETS][LOCAL_BUFFER_SIZE];

__host uint32_t vec_size;
__host uint32_t op;

static int alignUpTo2(int value) { return (value + 1) & ~1; }
static int alignUpTo8(int value) { return (value + 7) & ~7; }

// Find maximum
__host float max;
__dma_aligned float local_max[NR_TASKLETS];
BARRIER_INIT(max_reduce_barrier, NR_TASKLETS);

static void f_max(int tasklet_id, float *vec_mram, int elems_per_tasklet) {
  local_max[tasklet_id] = FLT_MIN;
  for (int i = 0; i < elems_per_tasklet; i += LOCAL_BUFFER_SIZE) {
    int num_elems = LOCAL_BUFFER_SIZE;
    unsigned int buffer_size = num_elems * sizeof(float);
    if (i + LOCAL_BUFFER_SIZE > elems_per_tasklet) {
      num_elems = elems_per_tasklet - i;
      buffer_size = alignUpTo8(num_elems * sizeof(float));
    }

    mram_read((__mram_ptr void *)(vec_mram + i), vec_local[tasklet_id], buffer_size);
    for (int j = 0; j < num_elems; j++) {
      if (vec_local[tasklet_id][j] > local_max[tasklet_id]) {
        local_max[tasklet_id] = vec_local[tasklet_id][j];
      }
    }
  }

  barrier_wait(&max_reduce_barrier);
  if (tasklet_id == 0) {
    max = FLT_MIN;
    for (int i = 0; i < NR_TASKLETS; i++) {
      if (local_max[i] > max) {
        max = local_max[i];
      }
    }
  }
}

// Exponentiate and calculate local sum
__host float sum;
__host float global_max;
__dma_aligned float local_sum[NR_TASKLETS];
BARRIER_INIT(sum_reduce_barrier, NR_TASKLETS);

// src: http://www.machinedlearnings.com/2011/06/fast-approximate-logarithm-exponential.html
inline float fastpow2(float p) {
  union {
    float f;
    uint32_t i;
  } vp = {p};
  int sign = (vp.i >> 31);
  int w = p;
  float z = p - w + sign;
  union {
    uint32_t i;
    float f;
  } v = {(1 << 23) * (p + 121.2740838f + 27.7280233f / (4.84252568f - z) - 1.49012907f * z)};  // What the fuck?
  return v.f;
}

inline float fastexp(float p) { return fastpow2(1.442695040f * p); }

void f_exp_and_calc_sum(int tasklet_id, float *vec_mram, int elems_per_tasklet) {
  local_sum[tasklet_id] = 0.0f;
  for (int i = 0; i < elems_per_tasklet; i += LOCAL_BUFFER_SIZE) {
    int num_elems = LOCAL_BUFFER_SIZE;
    unsigned int buffer_size = num_elems * sizeof(float);
    if (i + LOCAL_BUFFER_SIZE > elems_per_tasklet) {
      num_elems = elems_per_tasklet - i;
      buffer_size = alignUpTo8(num_elems * sizeof(float));
    }

    mram_read((__mram_ptr void *)(vec_mram + i), vec_local[tasklet_id], buffer_size);
    for (int j = 0; j < num_elems; j++) {
      // e^(xi - xmax)
      vec_local[tasklet_id][j] = fastexp(vec_local[tasklet_id][j] - global_max);
      local_sum[tasklet_id] += vec_local[tasklet_id][j];
    }
    // Write back the new values
    mram_write(vec_local[tasklet_id], (__mram_ptr void *)(vec_mram + i), buffer_size);
  }

  barrier_wait(&sum_reduce_barrier);
  if (tasklet_id == 0) {
    sum = 0.0f;
    for (int i = 0; i < NR_TASKLETS; i++) {
      sum += local_sum[i];
    }
  }
}

// Divide by the global sum
__host float divisor;
void f_divide(int tasklet_id, float *vec_mram, int elems_per_tasklet) {
  for (int i = 0; i < elems_per_tasklet; i += LOCAL_BUFFER_SIZE) {
    int num_elems = LOCAL_BUFFER_SIZE;
    unsigned int buffer_size = num_elems * sizeof(float);
    if (i + LOCAL_BUFFER_SIZE > elems_per_tasklet) {
      num_elems = elems_per_tasklet - i;
      buffer_size = alignUpTo8(num_elems * sizeof(float));
    }

    mram_read((__mram_ptr void *)(vec_mram + i), vec_local[tasklet_id], buffer_size);
    for (int j = 0; j < num_elems; j++) {
      vec_local[tasklet_id][j] /= divisor;
    }
    // Write back the new values
    mram_write(vec_local[tasklet_id], (__mram_ptr void *)(vec_mram + i), buffer_size);
  }
}

int main() {
  int tasklet_id = me();

  int elems_per_tasklet = alignUpTo2((vec_size - 1) / NR_TASKLETS + 1);
  float *vec_mram = (float *)(DPU_MRAM_HEAP_POINTER + tasklet_id * elems_per_tasklet * sizeof(float));

  int elems_up_to_tasklet = (tasklet_id)*elems_per_tasklet;
  if (elems_up_to_tasklet + elems_per_tasklet > vec_size) {
    elems_per_tasklet = vec_size - elems_up_to_tasklet;
  }

  if (op == 0) {
    f_max(tasklet_id, vec_mram, elems_per_tasklet);
  } else if (op == 1) {
    f_exp_and_calc_sum(tasklet_id, vec_mram, elems_per_tasklet);
  } else if (op == 2) {
    f_divide(tasklet_id, vec_mram, elems_per_tasklet);
  }

  return 0;
}