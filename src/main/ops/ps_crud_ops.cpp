#include "ps_ops.h"

namespace tensorflow {

REGISTER_OP("GetPSHandle")
    .Output("byte_ps_shard: Ref(string)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("use_node_name_sharing: bool = false")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .SetIsStateful()
    .SetShapeFn(TwoElementOutput);

REGISTER_OP("PSPull")
    .Input("byte_ps_shard: Ref(string)")
    .Input("keys: Tin")
    .Input("default_value: Tout")
    .Output("values: Tout")
    .Attr("Tin: type")
    .Attr("Tout: type")
    .SetShapeFn([](InferenceContext *c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));

      ShapeAndType value_shape_and_type;
      TF_RETURN_IF_ERROR(ValidateResourceHandle(c,
                                                /*keys=*/c->input(1),
                                                /*key_dtype_attr=*/"Tin",
                                                /*value_dtype_attr=*/"Tout",
                                                /*is_lookup=*/true,
                                                &value_shape_and_type));
      c->set_output(0, value_shape_and_type.shape);

      return Status::OK();
    });

REGISTER_OP("PSPush")
    .Input("byte_ps_shard: Ref(string)")
    .Input("keys: Tin")
    .Input("values: Tout")
    .Attr("Tin: type")
    .Attr("Tout: type")
    .SetShapeFn([](InferenceContext *c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));

      // TODO(ebrevdo): Validate keys and values shape.
      return Status::OK();
    });

REGISTER_OP("PSLoad")
    .Input("byte_ps_shard: Ref(string)")
    .Input("keys: Tin")
    .Input("values: Tout")
    .Attr("Tin: type")
    .Attr("Tout: type")
    .SetShapeFn([](InferenceContext *c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));

      // TODO(ebrevdo): Validate keys and values shape.
      return Status::OK();
    });

REGISTER_OP("PSSave")
    .Input("byte_ps_shard: Ref(string)")
    .Output("keys: Tkeys")
    .Output("values: Tvalues")
    .Attr("Tkeys: type")
    .Attr("Tvalues: type")
    .SetShapeFn([](InferenceContext *c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));

      ShapeHandle values = c->UnknownShape();
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(values, 1, &values));
      ShapeHandle keys = c->Vector(c->Dim(values, 0));
      c->set_output(0, keys);
      c->set_output(1, values);
      return Status::OK();
    });

} // namespace tensorflow
