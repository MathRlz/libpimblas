#include <mram.h>
#include <barrier.h>
#include <defs.h>
#include <stdio.h>

__host uint32_t vec_size;
__host float max;

#define FLT_MIN 1.175494351e-38F
#define LOCAL_BUFFER_SIZE 512
__dma_aligned float vec_local[NR_TASKLETS][LOCAL_BUFFER_SIZE];
__dma_aligned float local_max[NR_TASKLETS];


BARRIER_INIT(max_reduce_barrier, NR_TASKLETS);

static int alignUpTo2(int value) { return (value + 1) & ~1; }

int main() {
    int tasklet_id = me();

    int elems_per_tasklet = alignUpTo2((vec_size - 1) / NR_TASKLETS + 1);
    float *vec_mram = (float *)(DPU_MRAM_HEAP_POINTER + tasklet_id * elems_per_tasklet * sizeof(float));

    int elems_up_to_tasklet = (tasklet_id) * elems_per_tasklet;
    if (elems_up_to_tasklet + elems_per_tasklet > vec_size) {
        elems_per_tasklet = vec_size - elems_up_to_tasklet;
    }

    local_max[tasklet_id] = FLT_MIN;
    for (int i = 0; i < elems_per_tasklet; i += LOCAL_BUFFER_SIZE) {
        
        int num_elems = LOCAL_BUFFER_SIZE;
        if (i + LOCAL_BUFFER_SIZE > elems_per_tasklet) {
            num_elems = elems_per_tasklet - i;
        }

        mram_read((__mram_ptr void *)(vec_mram + i), vec_local[tasklet_id], sizeof(float) * LOCAL_BUFFER_SIZE);
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

    return 0;
}