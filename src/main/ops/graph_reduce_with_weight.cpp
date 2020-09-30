#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"


namespace tensorflow {

    const std::string op_doc = R"doc(
Kernel implementation of Dynamic Partition Softmax,
which allow to calculate softmax alone Dynamic Partition.

src_tensor:        the feature of src, must be 1-D, or 2-D tensor
dst_tensor:        the feature of dst, must be 1-D, or 2-D tensor
adj_tensor:        adj_tensor, must be 2-D int32 tensor
weight_tensor:
graph_reduce_with_weight_out:
)doc";

    REGISTER_OP("GraphReduceWithWeight")
            .Attr("reduce_method: string = 'mean'")
            .Input("src_tensor: float32")
            .Input("dst_tensor: float32")
            .Input("adj_tensor: int32")
            .Input("weight_tensor: float32")
            .Output("graph_reduce_with_weight_out: float32")
            .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
                c->set_output(0, c->input(0));
                return Status::OK();
            });

} //tensorflow