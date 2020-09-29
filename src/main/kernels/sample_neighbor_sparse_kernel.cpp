#include <memory>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "euler/client/graph.h"

using namespace tensorflow;

extern std::unique_ptr<euler::client::Graph> &Graph();

class SampleNeighborSparseOp : public AsyncOpKernel {
private:
    int count_;
    int default_node_;

public:
    explicit SampleNeighborSparseOp(OpKernelConstruction *ctx) : AsyncOpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("count", &count_));
    }

    void ComputeAsync(OpKernelContext *ctx, DoneCallback done) override;
};

void SampleNeighborSparseOp::ComputeAsync(OpKernelContext *ctx, DoneCallback done) {
    auto nodes = ctx->input(0);
    auto edge_types = ctx->input(1);

    auto nodes_flat = nodes.flat<int64>();
    std::vector<euler::client::NodeID> node_ids(nodes_flat.size());
    for (int i = 0; i < nodes_flat.size(); ++i) {
        node_ids[i] = nodes_flat(i);
    }

    auto etypes_flat = edge_types.flat<int32>();
    std::vector<int> etypes(etypes_flat.size());
    for (int i = 0; i < etypes_flat.size(); ++i) {
        etypes[i] = etypes_flat(i);
    }

    std::vector<int64> indices_col1, indices_col2;
    std::vector<int64> neighbors;
    std::vector<float> weights;
    std::vector<int32> types;

    auto callback = [&indices_col1, &indices_col2, &neighbors, &weights, &types, done, this](
            const euler::client::IDWeightPairVec &result) {
        for (size_t i = 0; i < result.size(); ++i) {
            for (size_t j = 0; j < result[i].size(); ++j) {
                neighbors.push_back(std::get<0>(result[i][j]));
                weights.push_back(std::get<1>(result[i][j]));
                types.push_back(std::get<2>(result[i][j]));
                indices_col1.push_back(i);
                indices_col2.push_back(j);
            }
        }
        done();
    };

    Graph()->SampleNeighbor(node_ids, etypes, count_, callback);

    Tensor* indices = nullptr;
    Tensor* neighbors_value = nullptr;
    Tensor* weights_value = nullptr;
    Tensor* types_value = nullptr;
    TensorShape index_shape;
    output_shape.AddDim(indices_col1.size());
    output_shape.AddDim(2);
    TensorShape value_shape;
    output_shape.AddDim(indices_col1.size());

    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, index_shape, &indices));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, value_shape, &neighbors_value));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, value_shape, &weights_value));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(3, value_shape, &types_value));

    auto indices_matrix = indices.matrix<int64>();
    auto neighbors_flat = neighbors_value->flat<int64>();
    auto weights_flat = weights_value->flat<float>();
    auto types_flat = types_value->flat<int32>();
    ctx->SetAttr()

    for (int i=0; i< indices_col1.size(); i++) {
        indices_matrix(i, 0) = indices_col1[i];
        indices_matrix(i, 1) = indices_col2[i];
        neighbors_flat(i) = neighbors[i];
        weights_flat(i) = weights[i];
        types_flat(i) = types[i];
    }
}

REGISTER_KERNEL_BUILDER(
        Name("SampleNeighborSparse").Device(DEVICE_CPU), SampleNeighborSparseOp);

// namespace tensorflow
