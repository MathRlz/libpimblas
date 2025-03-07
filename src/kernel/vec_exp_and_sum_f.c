#include <mram.h>
#include <barrier.h>
#include <defs.h>
#include <stdio.h>

__host uint32_t vec_size;
__host float max;
__host float sum;

#define LOCAL_BUFFER_SIZE 512
__dma_aligned float vec_local[NR_TASKLETS][LOCAL_BUFFER_SIZE];
__dma_aligned float local_sum[NR_TASKLETS];

BARRIER_INIT(sum_reduce_barrier, NR_TASKLETS);

// Polynomial coefficients for exp(x) in [0, ln(2)]
float exp_poly_coeffs[] = {
    1.0f, 1.0f, 0.5f, 0.1666667f, 0.0416667f
};

// Function to compute expf using range reduction and polynomial approximation
float expf_range_reduction(float x) {
    const float ln2 = 0.69314718056f; // ln(2)
    if (x < 0.0f) {
        return 1.0f / expf_range_reduction(-x); // Handle negative x
    }

    // Reduce x to the range [0, ln(2)]
    int k = (int)(x / ln2);
    float reduced_x = x - k * ln2;

    // Compute exp(reduced_x) using a polynomial approximation
    float result = 0.0f;
    float power = 1.0f;
    for (int i = 0; i < 5; i++) {
        result += exp_poly_coeffs[i] * power;
        power *= reduced_x;
    }

    // Multiply by 2^k to get the final result
    return result * (1 << k);
}

static inline float expf_taylor(float x) {
  return 1.0f + x + (x * x) / 2.0f + (x * x * x) / 6.0f + (x * x * x * x) / 24.0f;
}

static inline float expf_fast(float x) {
   union { float f; int i; } y;
   //y.i = (int)(x * 0xB5645F + 0x3F7893F5);
   y.i = (int)(x * 12102203.0f + 1064866805.0f);
   return (y.f);
}

static float expf(float x) {
    if (x >= -10.f && x <= 10.0f) {
        return expf_fast(x);
    }
    return expf_taylor(x);
}

static int alignUpTo2(int value) { return (value + 1) & ~1; }
static int alignUpTo8(int value) { return (value + 7) & ~7; }

int main() {
    int tasklet_id = me();

    int elems_per_tasklet = alignUpTo2((vec_size - 1) / NR_TASKLETS + 1);
    float *vec_mram = (float *)(DPU_MRAM_HEAP_POINTER + tasklet_id * elems_per_tasklet * sizeof(float));

    int elems_up_to_tasklet = (tasklet_id) * elems_per_tasklet;
    if (elems_up_to_tasklet + elems_per_tasklet > vec_size) {
        elems_per_tasklet = vec_size - elems_up_to_tasklet;
    }

    local_sum[tasklet_id] = 0.0f;
    for (int i = 0; i < elems_per_tasklet; i += LOCAL_BUFFER_SIZE) {
        
        int num_elems = LOCAL_BUFFER_SIZE;
        unsigned int buffer_size =  num_elems * sizeof(float);
        if (i + LOCAL_BUFFER_SIZE > elems_per_tasklet) {
            num_elems = elems_per_tasklet - i;
            buffer_size = alignUpTo8(num_elems * sizeof(float));
        }


        mram_read((__mram_ptr void *)(vec_mram + i), vec_local[tasklet_id], buffer_size);
        for (int j = 0; j < num_elems; j++) {
            // e^(xi - xmax)
            vec_local[tasklet_id][j] = expf_range_reduction(vec_local[tasklet_id][j] - max);
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

    return 0;
}