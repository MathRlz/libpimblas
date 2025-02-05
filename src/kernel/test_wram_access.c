#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <perfcounter.h>

struct params {
  uint32_t no_tasklets;
  uint32_t num_iters;
  int int_val;
  float float_val;
};

__host struct params args;

__host perfcounter_pair_t read_and_add_int_results[NR_TASKLETS];
__host perfcounter_pair_t write_and_add_int_results[NR_TASKLETS];
__host perfcounter_pair_t rw_and_add_int_results[NR_TASKLETS];

__host perfcounter_pair_t read_and_add_float_results[NR_TASKLETS];
__host perfcounter_pair_t write_and_add_float_results[NR_TASKLETS];
__host perfcounter_pair_t rw_and_add_float_results[NR_TASKLETS];

inline void start_count() { perfcounter_config(COUNT_ENABLE_BOTH, true); }

inline void end_count(perfcounter_pair_t *counters) { counters[me()] = perfcounter_get_both(false); }

BARRIER_INIT(mem_reset_barrier, NR_TASKLETS);

void read_and_add_int(int *buffer, int buffer_len) {
  volatile int a = 0;
  for (int i = 0; i < buffer_len; i++) {
    a += buffer[i];
  }
}

void rw_and_add_int(int *buffer, int buffer_len) {
  volatile int a = 0;
  for (int i = 0; i < buffer_len; i++) {
    a += buffer[i];
    buffer[i] = a;
  }
}

void write_and_add_int(int *buffer, int val, int buffer_len) {
  volatile int a = 0;
  for (int i = 0; i < buffer_len; i++) {
    a += val;
    buffer[i] = a;
  }
}

void read_and_add_float(float *buffer, int buffer_len) {
  volatile float a = 0;
  for (int i = 0; i < buffer_len; i++) {
    a += buffer[i];
  }
}

void rw_and_add_float(float *buffer, int buffer_len) {
  volatile float a = 0;
  for (int i = 0; i < buffer_len; i++) {
    a += buffer[i];
    buffer[i] = a;
  }
}

void write_and_add_float(float *buffer, int val, int buffer_len) {
  volatile float a = 0;
  for (int i = 0; i < buffer_len; i++) {
    a += val;
    buffer[i] = a;
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

  int *buffer_int = (int *)mem_alloc(args.num_iters * sizeof(int));
  for (int i = 0; i < args.num_iters; i++) {
    buffer_int[i] = args.int_val;
  }

  start_count();
  read_and_add_int(buffer_int, args.num_iters);
  end_count(read_and_add_int_results);

  start_count();
  rw_and_add_int(buffer_int, args.num_iters);
  end_count(rw_and_add_int_results);

  for (int i = 0; i < args.num_iters; i++) {
    buffer_int[i] = args.int_val;
  }

  start_count();
  write_and_add_int(buffer_int, args.int_val, args.num_iters);
  end_count(write_and_add_int_results);

  float *buffer_float = (float *)buffer_int;
  for (int i = 0; i < args.num_iters; i++) {
    buffer_float[i] = args.float_val;
  }

  start_count();
  read_and_add_float(buffer_float, args.num_iters);
  end_count(read_and_add_float_results);

  start_count();
  rw_and_add_float(buffer_float, args.num_iters);
  end_count(rw_and_add_float_results);

  for (int i = 0; i < args.num_iters; i++) {
    buffer_float[i] = args.float_val;
  }

  start_count();
  write_and_add_float(buffer_float, args.float_val, args.num_iters);
  end_count(write_and_add_float_results);

  return 0;
}