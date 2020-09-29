#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"


namespace tensorflow {

    const std::string op_doc = R"doc(
Kernel implementation of Dynamic Partition Softmax,
which allow to calculate softmax alone Dynamic Partition.

grad_tensor:        input_tensor, must be 1-D, or [None, 1] tensor
adj_tensor:        adj_tensor, must be 2-D int32 tensor
)doc";

    REGISTER_OP("GraphReduceGrad")
            .Attr("reduce_method: string = 'mean'")
            .Input("grad_tensor: float32")
            .Input("adj_tensor: int32")
            .Output("graph_reduce_grad_out: float32")
            .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
                c->input_tensor(0)->dim_size(1);
                c->set_output(0, c->input(0));
                return Status::OK();
            });

} //tensorflow