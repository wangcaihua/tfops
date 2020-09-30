#include <vector>
#include <iostream>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"


namespace tensorflow {

    const std::string op_doc = R"doc(
Kernel implementation of Dynamic Partition Softmax,
which allow to calculate softmax alone Dynamic Partition.

input_list:        the feature of src, must be 1-D, or 2-D tensor
unique_nodes:        the feature of dst, must be 1-D, or 2-D tensor
indices:        adj_tensor, must be 2-D int32 tensor
)doc";

    REGISTER_OP("GraphReorder")
            .Attr("N: int")
            .Input("input_list: N * int64")
            .Output("unique_nodes: int64")
            .Output("indices: N * int32")
            .SetShapeFn(shape_inference::UnknownShape);

} //tensorflow