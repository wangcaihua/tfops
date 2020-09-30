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

        assert(dst_tensor.shape().dim_size(0) <= adj_tensor.shape().dim_size(0));

        const int num_src = src_tensor.dim_size(0);
        const int num_edge = adj_tensor.dim_size(0);
        // const int num_dst = dst_tensor.dim_size(0);

        Tensor *output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, src_tensor.shape(), &output_tensor));
        auto adjs = adj_tensor.matrix<int32>();

        if (dst_tensor.dims() == 1) {
            auto dst_flat = dst_tensor.flat<float>();
            auto output_flat = output_tensor->flat<float>();
            for (int i = 0; i < num_src; i++) {
                output_flat(i) = 0.0f;
            }

            if (reduce_method == "sum") {
                for (int i = 0; i < num_edge; i++) {
                    output_flat(adjs(i, 0)) += dst_flat(adjs(i, 1));
                }
            } else if (reduce_method == "mean") {
                int counts[num_src];
                memset(counts, 0, sizeof(counts));
                for (int p = 0; p < num_edge; p++) {
                    counts[adjs(p, 0)] += 1;
                }

                for (int i = 0; i < num_edge; i++) {
                    int32 k = adjs(i, 0);
                    auto weight = (float) (1.0 / counts[k]);
                    output_flat(k) += weight * dst_flat(adjs(i, 1));
                }
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

            if (reduce_method == "sum") {
                for (int i = 0; i < num_edge; i++) {
                    int32 p = adjs(i, 0), q = adjs(i, 1);
                    for (int j = 0; j < feat_dims; j++) {
                        output_matrix(p, j) += dst_matrix(q, j);
                    }
                }
            } else if (reduce_method == "mean") {
                int counts[num_src];
                memset(counts, 0, sizeof(counts));
                for (int p = 0; p < num_edge; p++) {
                    counts[adjs(p, 0)] += 1;
                }

                for (int i = 0; i < num_edge; i++) {
                    int32 p = adjs(i, 0), q = adjs(i, 1);
                    auto weight = (float) (1.0 / counts[p]);
                    for (int j = 0; j < feat_dims; j++) {
                        output_matrix(p, j) += weight * dst_matrix(q, j);
                    }
                }
            }
        } else {
            throw "Only 1-D, 2-D tensor supported!";
        }
    }

};


REGISTER_KERNEL_BUILDER(Name("GraphReduce").Device(DEVICE_CPU), GraphReduceOp);
