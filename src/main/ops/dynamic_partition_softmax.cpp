#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
    const std::string op_doc = R"doc(
Kernel implementation of Dynamic Partition Softmax,
which allow to calculate softmax alone Dynamic Partition.

input_tensor:        input_tensor, must be 1-D, or [None, 1] tensor
indices:             Partition indices, the same shape with
)doc";

    REGISTER_OP("DynamicPartitionSoftmax")
            .Input("input_tensor: float32")
            .Input("indices: int32")
            .Output("softmax_out: float32")
            .Doc(op_doc)
            .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
                c->set_output(0, c->input(0));
                return Status::OK();
            });

} // tensorflow
