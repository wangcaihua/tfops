#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

class GraphReduceWithWeightGradOp : public OpKernel {

public:
    explicit GraphReduceWithWeightGradOp(OpKernelConstruction *context) : OpKernel(context) {}

    void Compute(OpKernelContext *context) override {
        // Grab the input tensor
        const Tensor &grad_tensor = context->input(0);
        const Tensor &dst_tensor = context->input(1);
        const Tensor &adj_tensor = context->input(2);
        const Tensor &weight_tensor = context->input(3);

        assert(adj_tensor.dim_size(0) == weight_tensor.dim_size(0));

        // const int num_src = grad_tensor.dim_size(0);
        const int num_dst = dst_tensor.dim_size(0);
        const int num_edge = adj_tensor.dim_size(0);

        Tensor *dst_grad_tensor = nullptr;
        Tensor *weight_grad_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, dst_tensor.shape(), &dst_grad_tensor));
        OP_REQUIRES_OK(context, context->allocate_output(1, weight_tensor.shape(), &weight_grad_tensor));

        auto weight_flat = weight_tensor.flat<float>();
        auto weight_grad_flat = weight_grad_tensor->flat<float>();

        auto adjs = adj_tensor.matrix<int32>();
        if (grad_tensor.dims() == 1) {
            auto src_grad_flat = grad_tensor.flat<float>();
            auto dst_flat = dst_tensor.flat<float>();
            auto dst_grad_flat = dst_grad_tensor->flat<float>();

            for (int i = 0; i < num_dst; i++) {
                dst_grad_flat(i) = 0.0f;
            }

            for (int i = 0; i < num_edge; i++) {
                int32 p = adjs(i, 0), q = adjs(i, 1);
                dst_grad_flat(q) += weight_flat(i) * src_grad_flat(p);
                weight_grad_flat(i) = src_grad_flat(p) * dst_flat(p);
            }
        } else if (grad_tensor.dims() == 2) {
            int64 feat_dims = grad_tensor.dim_size(1);
            OP_REQUIRES_OK(context, context->allocate_output(1, dst_tensor.shape(), &dst_grad_tensor));
            auto src_grad_matrix = grad_tensor.matrix<float>();
            auto dst_matrix = dst_tensor.matrix<float>();
            auto dst_grad_matrix = dst_grad_tensor->matrix<float>();

            for (int i = 0; i < num_dst; i++) {
                for (int j = 0; j < feat_dims; j++) {
                    dst_grad_matrix(i, j) = 0.0f;
                }
            }

            for (int i=0; i < num_edge; i++){
                weight_grad_flat(i) = 0.0f;
            }

            for (int i = 0; i < num_edge; i++) {
                int32 p = adjs(i, 0), q = adjs(i, 1);
                auto weight = weight_flat(i);
                for (int j=0; j< feat_dims; j++) {
                    dst_grad_matrix(q, j) += weight * src_grad_matrix(p, j);
                    weight_grad_flat(i) += src_grad_matrix(p, j) * dst_matrix(q, j);
                }
            }
        } else {
            throw "Only 1-D, 2-D tensor supported!";
        }
    }

};


REGISTER_KERNEL_BUILDER(Name("GraphReduceWithWeightGrad").Device(DEVICE_CPU), GraphReduceWithWeightGradOp);