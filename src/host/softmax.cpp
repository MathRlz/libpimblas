#include "common.hpp"
#include "kernel.hpp"

extern "C" {

    int softmax(const float *vec_in, float *vec_out, size_t size) {
        Kernel softmax;

        size_t chunk_size = size;
        uint32_t nr_dpus = 1;
        std::cout << "nr dpus: " << nr_dpus << std::endl;
        if (false == softmax.allocate_n(nr_dpus)) {
            return -1;
        }

        // First get maximum in the whole vec
        softmax.load_program("vec_max_f.kernel");

        // Load vector into DPUs
        softmax.set_arg_scatter(DPU_MRAM_HEAP_POINTER_NAME, 0, vec_in, chunk_size * sizeof(float), size * sizeof(float), false);

        uint32_t vec_size = size;
        softmax.set_arg_broadcast_exact("vec_size", 0, &vec_size, sizeof(uint32_t), false);
        softmax.launch(false);
        
        std::vector<float> max_values(nr_dpus);
        softmax.get_arg_copy_each("max", 0, max_values.data(), sizeof(float));

        float max_val = std::numeric_limits<float>::min();
        for (auto val : max_values) {
            if (val > max_val) {
                max_val = val;
            }
        }
        std::cout << "max_val = " << max_val << std::endl;

        //softmax.read_log();

        // Compute exponentials and local sum
        softmax.load_program("vec_exp_and_sum_f.kernel");

        softmax.set_arg_broadcast_exact("vec_size", 0, &vec_size, sizeof(float), false);
        softmax.set_arg_broadcast_exact("max", 0, &max_val, sizeof(float), false);

        softmax.launch(false);
        std::vector<float> sums(nr_dpus);
        softmax.get_arg_copy_each("sum", 0, sums.data(), sizeof(float));

        float sum = 0.0f;
        for (auto val : sums) {
            sum += val;
        }
        std::cout << "sum = " << sum << std::endl;

        // Compute final softmax value by diving by the global sum
        softmax.load_program("vec_divide_f.kernel");

        softmax.set_arg_broadcast_exact("vec_size", 0, &vec_size, sizeof(float), false);
        softmax.set_arg_broadcast_exact("divisor", 0, &sum, sizeof(float), false);
        softmax.launch(false);

        softmax.get_arg_gather(DPU_MRAM_HEAP_POINTER_NAME, 0, reinterpret_cast<void*>(vec_out), chunk_size * sizeof(float), size * sizeof(float), false);

        return 0;
    }
}