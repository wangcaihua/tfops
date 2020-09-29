#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

class GraphReduceOp : public OpKernel {
private:
    std::string reduce_method = "mean";

public:
    explicit GraphReduceOp(OpKernelConstruction *context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("reduce_method", &reduce_method));
    }

    void Compute(OpKernelContext *context) override {
        // Grab the input tensor
        const Tensor &src_tensor = context->input(0);
        const Tensor &dst_tensor = context->input(1);
        const Tensor &adj_tensor = context->input(2);

        assert(dst_tensor.shape().dim_size(0) == adj_tensor.shape().dim_size(0));

        auto max_len = src_tensor.dim_size(0);
        auto &shape = dst_tensor.shape();
        const int N = shape.dim_size(0);
        auto output_shape = TensorShape();
        output_shape.AddDim(max_len);

        Tensor *output_tensor = nullptr;
        auto adjs = adj_tensor.matrix<int32>();

        if (shape.dims() == 1) {
            auto input_flat = dst_tensor.flat<float>();
            OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
            auto output_flat = output_tensor->flat<float>();
            for (int i = 0; i < N; i++) {
                output_flat(i) = 0.0f;
            }

            if (reduce_method == "sum") {
                for (int i = 0; i < N; i++) {
                    output_flat(adjs(i, 0)) += input_flat(i);
                }
            } else if (reduce_method == "mean") {
                int counts[max_len];
                memset(counts, 0, sizeof(counts));
                for (int i = 0; i < N; i++) {
                    counts[adjs(i, 0)] += 1;
                }

                for (int i = 0; i < N; i++) {
                    int32 k = adjs(i, 0);
                    auto weight = (float) (1.0 / counts[k]);
                    output_flat(k) += weight * input_flat(i);
                }
            }
        } else if (shape.dims() == 2) {
            auto M = shape.dim_size(1);
            output_shape.AddDim(M);
            OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));

            auto input_matrix = dst_tensor.matrix<float>();
            auto output_matrix = output_tensor->matrix<float>();
            for (int i = 0; i < max_len; i++) {
                for (int j=0; j < M; j++) {
                    output_matrix(i, j) = 0.0f;
                }
            }

            if (reduce_method == "sum") {
                for (int i = 0; i < N; i++) {
                    int32 k = adjs(i, 0);
                    for (int j = 0; j < M; j++) {
                        output_matrix(k, j) += input_matrix(i, j);
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
                    auto weight = (float) (1.0 / counts[k]);
                    for (int j = 0; j < M; j++) {
                        output_matrix(k, j) += weight * input_matrix(i, j);
                    }
                }
            }
        } else {
            throw "Only 1-D, 2-D tensor supported!";
        }

        context->set_output(0, *output_tensor);
    }

};


REGISTER_KERNEL_BUILDER(Name("GraphReduce").Device(DEVICE_CPU), GraphReduceOp);
