#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

class GraphReduceGradOp : public OpKernel {
private:
    std::string reduce_method = "mean";

public:
    explicit GraphReduceGradOp(OpKernelConstruction *context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("reduce_method", &reduce_method));
    }

    void Compute(OpKernelContext *context) override {
        // Grab the input tensor
        const Tensor &grad_tensor = context->input(0);
        const Tensor &adj_tensor = context->input(1);

        const int N = adj_tensor.dim_size(0);
        const auto max_len = grad_tensor.dim_size(0);
        auto output_shape = TensorShape();
        output_shape.AddDim(N);

        Tensor *output_tensor = nullptr;
        auto adjs = adj_tensor.matrix<int32>();
        if (grad_tensor.dims() == 1) {
            OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
            auto output_flat = output_tensor->flat<float>();
            auto grad_flat = grad_tensor.flat<float>();
            if (reduce_method == "sum") {
                for (int i = 0; i < N; i++) {
                    output_flat(i) = grad_flat(adjs(i, 0));
                }
            } else if (reduce_method == "mean") {
                int counts[max_len];
                memset(counts, 0, sizeof(counts));
                for (int i = 0; i < N; i++) {
                    counts[adjs(i, 0)] += 1;
                }

                for (int i = 0; i < N; i++) {
                    int32 k = adjs(i, 0);
                    auto weight = (float)(1.0 / counts[k]);
                    output_flat(i) = weight * grad_flat(k);
                }
            }
        } else if (grad_tensor.dims() == 2) {
            int64 M = grad_tensor.dim_size(1);
            output_shape.AddDim(M);
            OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
            auto output_matrix = output_tensor->matrix<float>();
            auto grad_matrix = grad_tensor.matrix<float>();

            if (reduce_method == "sum") {
                for (int i = 0; i < N; i++) {
                    for (int j=0; j< M; j++) {
                        output_matrix(i, j) =  grad_matrix(adjs(i, 0), j);
                    }
                }
            } else if (reduce_method == "mean") {
                int counts[max_len];
                memset(counts, 0, sizeof(counts));
                for (int i = 0; i < N; i++) {
                    counts[adjs(i, 0)] += 1;
                }

                for (int i = 0; i < N; i++) {
                    int32 k = adjs(i, 0);
                    auto weight = (float)(1.0 / counts[k]);
                    for (int j=0; j< M; j++) {
                        output_matrix(i, j) = weight * grad_matrix(k, j);
                    }
                }
            }
        } else {
            throw "Only 1-D, 2-D tensor supported!";
        }

        context->set_output(0, *output_tensor);
    }

};


REGISTER_KERNEL_BUILDER(Name("GraphReduceGrad").Device(DEVICE_CPU), GraphReduceGradOp);

