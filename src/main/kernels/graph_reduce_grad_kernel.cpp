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
        const Tensor &dst_tensor = context->input(1);
        const Tensor &adj_tensor = context->input(2);

        const int num_src = grad_tensor.dim_size(0);
        const int num_dst = dst_tensor.dim_size(0);
        const int num_edge = adj_tensor.dim_size(0);

        auto output_shape = TensorShape();
        output_shape.AddDim(num_dst);

        Tensor *output_tensor = nullptr;
        auto adjs = adj_tensor.matrix<int32>();
        if (grad_tensor.dims() == 1) {
            OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
            auto output_flat = output_tensor->flat<float>();
            auto grad_flat = grad_tensor.flat<float>();
            for (int i = 0; i < num_dst; i++) {
                output_flat(i) = 0.0f;
            }

            if (reduce_method == "sum") {
                for (int i = 0; i < num_edge; i++) {
                    output_flat(adjs(i, 1)) += grad_flat(adjs(i, 0));
                }
            } else if (reduce_method == "mean") {
                int counts[num_src];
                memset(counts, 0, sizeof(counts));
                for (int i = 0; i < num_edge; i++) {
                    counts[adjs(i, 0)] += 1;
                }

                for (int i = 0; i < num_edge; i++) {
                    int32 k = adjs(i, 0);
                    auto weight = (float)(1.0 / counts[k]);
                    output_flat(adjs(i, 1)) += weight * grad_flat(k);
                }
            }
        } else if (grad_tensor.dims() == 2) {
            int64 feat_dims = grad_tensor.dim_size(1);
            output_shape.AddDim(feat_dims);
            OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
            auto output_matrix = output_tensor->matrix<float>();
            auto grad_matrix = grad_tensor.matrix<float>();
            for (int i = 0; i < num_dst; i++) {
                for (int j = 0; j < feat_dims; j++) {
                    output_matrix(i, j) = 0.0f;
                }
            }

            if (reduce_method == "sum") {
                for (int i = 0; i < num_edge; i++) {
                    int32 p = adjs(i, 0), q = adjs(i, 1);
                    for (int j=0; j< feat_dims; j++) {
                        output_matrix(q, j) += grad_matrix(p, j);
                    }
                }
            } else if (reduce_method == "mean") {
                int counts[num_src];
                memset(counts, 0, sizeof(counts));
                for (int i = 0; i < num_edge; i++) {
                    counts[adjs(i, 0)] += 1;
                }

                for (int i = 0; i < num_edge; i++) {
                    int32 p = adjs(i, 0), q = adjs(i, 1);
                    auto weight = (float)(1.0 / counts[p]);
                    for (int j=0; j< feat_dims; j++) {
                        output_matrix(q, j) += weight * grad_matrix(p, j);
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

