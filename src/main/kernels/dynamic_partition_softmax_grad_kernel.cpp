#include <math.h>
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

class DynamicPartitionSoftmaxGradOp : public OpKernel {
public:
    explicit DynamicPartitionSoftmaxGradOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        // Grab the input tensor
        const Tensor& input_tensor = context->input(0);
        const Tensor& grad_tensor = context->input(1);
        const Tensor& indices_tensor = context->input(2);

        auto inputs = input_tensor.flat<float>();
        auto grads = grad_tensor.flat<float>();
        auto indices = indices_tensor.flat<int32>();

        int32 min_value= indices(0), max_value = indices(0);
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
        assert(min_value >=0);

        float grad_sum[max_value+1];
        memset(grad_sum, 0, sizeof(float) * (max_value + 1));
        for (int i = 0; i < N; i++) {
            grad_sum[indices(i)] += inputs(i) * grads(i);
        }

        // Create an output tensor
        Tensor* output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                         &output_tensor));
        auto outputs = output_tensor->flat<float>();

        // Set all but the first element of the output tensor to 0.
        for (int i = 0; i < N; i++) {
            outputs(i) = (grads(i) - grad_sum[indices(i)]) * inputs(i);
        }

        context->set_output(0, *output_tensor);
    }
};


REGISTER_KERNEL_BUILDER(Name("DynamicPartitionSoftmaxGrad").Device(DEVICE_CPU), DynamicPartitionSoftmaxGradOp);
