#include <vector>
#include <iostream>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"


namespace tensorflow {
    const std::string op_doc = R"doc(
GetTopKNeighborSparse

Sample Neighbors for nodes.

k: sample neighbor count for each node
nodes: Input, the nodes to sample neighbors for
edge_types: Input, the outing edge types to sample neighbors for
indices:
neighbors: Output, the sample result
weights:
types: default filling node if node has no neighbor

)doc";

    REGISTER_OP("GetTopKNeighborSparse")
            .Attr("k: int")
            .Input("nodes: int64")
            .Input("edge_types: int32")
            .SetIsStateful()
            .Output("indices: int64")
            .Output("neighbors: int64")
            .Output("weights: float")
            .Output("types: int32")
            .SetShapeFn(shape_inference::UnknownShape)
            .Doc(op_doc);

}