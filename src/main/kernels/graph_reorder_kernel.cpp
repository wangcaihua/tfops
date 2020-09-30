#include <vector>
#include <unordered_map>
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

class GraphReorderOp : public OpKernel {
private:
    int num_parts;

public:
    explicit GraphReorderOp(OpKernelConstruction *context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("N", &num_parts));
    }

    void Compute(OpKernelContext *context) override {
        // Grab the input tensor
        OpInputList op_inputs(context, 0, num_parts);
        OP_REQUIRES_OK(context, context->input_list("input_list", &op_inputs));

        std::unordered_map<int64, int> counts;
        std::vector<int64> unique_nodes;
        std::vector<std::vector<int> > indices;

        for (int i = 0; i < num_parts; i++) {
            const Tensor &part = op_inputs[i];
            auto part_flat = part.flat<int64>();
            std::vector<int> part_indices;

            for (int j=0; j < part_flat.size(); j++) {
                int64 node = part_flat(j);

                auto got = counts.find(node);
                int index;
                if (got == counts.end()) { // not found
                    index = counts.size();
                    counts.insert({node, index});
                    unique_nodes.push_back(node);
                } else {
                    index = got->second;
                }

                part_indices.push_back(index);
            }

            indices.push_back(part_indices);
        }

        Tensor *unique_nodes_tensor = nullptr;
        TensorShape unique_nodes_shape;
        unique_nodes_shape.AddDim(unique_nodes.size());
        OP_REQUIRES_OK(context, context->allocate_output(0, unique_nodes_shape, &unique_nodes_tensor));
        auto unique_nodes_flat = unique_nodes_tensor->flat<int64>();
        for (int i=0; i < unique_nodes.size(); i++){
            unique_nodes_flat(i) = unique_nodes[i];
        }

        OpOutputList op_output(context, 1, num_parts+1);
        for (int i=0; i < num_parts; i++) {
            std::vector<int> &part_indices = indices[i];
            int part_size = part_indices.size();
            Tensor *part_indices_tensor= nullptr;
            TensorShape part_shape;
            part_shape.AddDim(part_size);
            auto stat = op_output.allocate(i, part_shape, &part_indices_tensor);
            assert(stat == Status::OK());

            auto part_indices_flat = part_indices_tensor->flat<int32>();
            for(int j=0; j < part_size; j++) {
                part_indices_flat(j) = part_indices[j];
            }
        }
    }

};


REGISTER_KERNEL_BUILDER(Name("GraphReorder").Device(DEVICE_CPU), GraphReorderOp);
