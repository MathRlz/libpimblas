#include <defs.h>
#include <perfcounter.h>

struct params {
  uint32_t no_tasklets;
  uint32_t num_iters;
  int int_val;
  float float_val;
};

__host struct params args;

__host perfcounter_pair_t addition_int_results[NR_TASKLETS];
__host perfcounter_pair_t multiplication_int_results[NR_TASKLETS];
__host perfcounter_pair_t division_int_results[NR_TASKLETS];

__host perfcounter_pair_t addition_float_results[NR_TASKLETS];
__host perfcounter_pair_t multiplication_float_results[NR_TASKLETS];
__host perfcounter_pair_t division_float_results[NR_TASKLETS];

inline void start_count() { perfcounter_config(COUNT_ENABLE_BOTH, true); }

inline void end_count(perfcounter_pair_t *counters) { counters[me()] = perfcounter_get_both(false); }

void addition_int(int iters, int val) {
  volatile int a = 0;
  for (int i = 0; i < iters; i++) {
    a += val;
  }
}

void multiplication_int(int iters, int val) {
  volatile int a = 0;
  for (int i = 0; i < iters; i++) {
    a *= val;
  }
}

void division_int(int iters, int val) {
  volatile int a = 0;
  for (int i = 0; i < iters; i++) {
    a /= val;
  }
}

void addition_float(int iters, float val) {
  volatile float a = 0;
  for (int i = 0; i < iters; i++) {
    a += val;
  }
}

void multiplication_float(int iters, float val) {
  volatile float a = 0;
  for (int i = 0; i < iters; i++) {
    a *= val;
  }
}

void division_float(int iters, float val) {
  volatile float a = 0;
  for (int i = 0; i < iters; i++) {
    a /= val;
  }
}

int main() {
  if (me() >= args.no_tasklets) {
    return 0;
  }

  start_count();
  addition_int(args.num_iters, args.int_val);
  end_count(addition_int_results);

  start_count();
  multiplication_int(args.num_iters, args.int_val);
  end_count(multiplication_int_results);

  start_count();
  division_int(args.num_iters, args.int_val);
  end_count(division_int_results);

  start_count();
  addition_float(args.num_iters, args.float_val);
  end_count(addition_float_results);

  start_count();
  multiplication_float(args.num_iters, args.float_val);
  end_count(multiplication_float_results);

  start_count();
  division_float(args.num_iters, args.float_val);
  end_count(division_float_results);

  return 0;
}