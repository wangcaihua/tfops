#include <math.h>
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

class DynamicPartitionSoftmaxOp : public OpKernel {
public:
    explicit DynamicPartitionSoftmaxOp(OpKernelConstruction *context) : OpKernel(context) {}

    void Compute(OpKernelContext *context) override {
        // Grab the input tensor
        const Tensor& input_tensor = context->input(0);
        const Tensor& indices_tensor = context->input(1);

        assert(input_tensor.shape().dim_size(0) == indices_tensor.shape().dim_size(0));

        auto inputs = input_tensor.flat<float>();
        auto indices = indices_tensor.flat<int32>();

        int32 min_value = indices(0), max_value = indices(0);
        const int N = indices.size();
        for (int i = 1; i < N; i++) {
            int32 curr = indices(i);

            if (curr < min_value) {
                min_value = curr;
            }

            if (curr > max_value) {
                max_value = curr;
            }
        }

        assert(min_value >= 0);

        // Create an output tensor
        Tensor *output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                         &output_tensor));
        auto outputs = output_tensor->flat<float>();

        // Set all but the first element of the output tensor to 0.
        float weight[max_value + 1];
        memset(weight, 0, sizeof(float) * (max_value + 1));
        for (int i = 0; i < N; i++) {
            auto exp_value = exp(inputs(i));
            weight[indices(i)] += exp_value;
            outputs(i) = exp_value;
        }

        for (int i = 0; i < N; i++) {
            outputs(i) /= weight[indices(i)];
        }

        context->set_output(0, *output_tensor);
    }
};


REGISTER_KERNEL_BUILDER(Name("DynamicPartitionSoftmax").Device(DEVICE_CPU), DynamicPartitionSoftmaxOp);
