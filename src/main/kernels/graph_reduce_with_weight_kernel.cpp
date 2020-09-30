#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

class GraphReduceWithWeightOp : public OpKernel {
public:
    explicit GraphReduceWithWeightOp(OpKernelConstruction *context) : OpKernel(context) {}

    void Compute(OpKernelContext *context) override {
        // Grab the input tensor
        const Tensor &src_tensor = context->input(0);
        const Tensor &dst_tensor = context->input(1);
        const Tensor &adj_tensor = context->input(2);
        const Tensor &weight_tensor = context->input(3);

        assert(adj_tensor.dim_size(0) == weight_tensor.dim_size(0));

        const int num_src = src_tensor.dim_size(0);
        // const int num_dst = dst_tensor.dim_size(0);
        const int num_edge = adj_tensor.dim_size(0);

        Tensor *output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, src_tensor.shape(), &output_tensor));

        auto adjs = adj_tensor.matrix<int32>();
        auto weights = weight_tensor.flat<float>();

        if (dst_tensor.dims() == 1) {
            auto dst_flat = dst_tensor.flat<float>();
            auto output_flat = output_tensor->flat<float>();
            for (int i = 0; i < num_src; i++) {
                output_flat(i) = 0.0f;
            }

            for (int i = 0; i < num_edge; i++) {
                output_flat(adjs(i, 0)) += weights(i) * dst_flat(adjs(i, 1));
            }
        } else if (dst_tensor.dims() == 2) {
            auto feat_dims = dst_tensor.dim_size(1);
            auto dst_matrix = dst_tensor.matrix<float>();
            auto output_matrix = output_tensor->matrix<float>();
            for (int i = 0; i < num_src; i++) {
                for (int j = 0; j < feat_dims; j++) {
                    output_matrix(i, j) = 0.0f;
                }
            }

            for (int i = 0; i < num_edge; i++) {
                float weight = weights(i);
                int32 p = adjs(i, 0), q = adjs(i, 1);
                for (int j = 0; j < feat_dims; j++) {
                    output_matrix(p, j) += weight * dst_matrix(q, j);
                }
            }
        } else {
            throw "Only 1-D, 2-D tensor supported!";
        }
    }

};


REGISTER_KERNEL_BUILDER(Name("GraphReduceWithWeight").Device(DEVICE_CPU), GraphReduceWithWeightOp);
