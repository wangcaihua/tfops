#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeAndType;
using shape_inference::ShapeHandle;

Status ValidateResourceHandle(InferenceContext *c, ShapeHandle keys,
                              const string &key_dtype_attr,
                              const string &value_dtype_attr, bool is_lookup,
                              ShapeAndType *output_shape_and_type) {
  auto *handle_data = c->input_handle_shapes_and_types(0);
  if (handle_data == nullptr || handle_data->size() != 2) {
    output_shape_and_type->shape = c->UnknownShape();
    output_shape_and_type->dtype = DT_INVALID;
  } else {
    const ShapeAndType &key_shape_and_type = (*handle_data)[0];
    const ShapeAndType &value_shape_and_type = (*handle_data)[1];
    DataType key_dtype;
    TF_RETURN_IF_ERROR(c->GetAttr(key_dtype_attr, &key_dtype));
    if (key_shape_and_type.dtype != key_dtype) {
      return errors::InvalidArgument("Trying to read value with wrong dtype. "
                                     "Expected ",
                                     DataTypeString(key_shape_and_type.dtype),
                                     " got ", DataTypeString(key_dtype));
    }
    DataType value_dtype;
    TF_RETURN_IF_ERROR(c->GetAttr(value_dtype_attr, &value_dtype));
    if (value_shape_and_type.dtype != value_dtype) {
      return errors::InvalidArgument("Trying to read value with wrong dtype. "
                                     "Expected ",
                                     DataTypeString(value_shape_and_type.dtype),
                                     " got ", DataTypeString(value_dtype));
    }
    output_shape_and_type->dtype = value_shape_and_type.dtype;

    if (is_lookup) {
      if (c->RankKnown(key_shape_and_type.shape) && c->RankKnown(keys)) {
        int keys_rank = c->Rank(keys);
        int key_suffix_rank = c->Rank(key_shape_and_type.shape);
        if (keys_rank < key_suffix_rank) {
          return errors::InvalidArgument(
              "Expected keys to have suffix ",
              c->DebugString(key_shape_and_type.shape),
              " but saw shape: ", c->DebugString(keys));
        }
        for (int d = 0; d < key_suffix_rank; d++) {
          // Ensure the suffix of keys match what's in the Table.
          DimensionHandle dim = c->Dim(key_shape_and_type.shape, d);
          TF_RETURN_IF_ERROR(
              c->ReplaceDim(keys, keys_rank - key_suffix_rank + d, dim, &keys));
        }
        std::vector<DimensionHandle> keys_prefix_vec;
        keys_prefix_vec.reserve(keys_rank - key_suffix_rank);
        for (int d = 0; d < keys_rank - key_suffix_rank; ++d) {
          keys_prefix_vec.push_back(c->Dim(keys, d));
        }
        ShapeHandle keys_prefix = c->MakeShape(keys_prefix_vec);
        TF_RETURN_IF_ERROR(c->Concatenate(keys_prefix,
                                          value_shape_and_type.shape,
                                          &output_shape_and_type->shape));
      } else {
        output_shape_and_type->shape = c->UnknownShape();
      }
    } else {
      TF_RETURN_IF_ERROR(c->Concatenate(keys, value_shape_and_type.shape,
                                        &output_shape_and_type->shape));
    }
  }
  return Status::OK();
}

Status TwoElementOutput(InferenceContext *c) {
  c->set_output(0, c->Vector(2));
  return Status::OK();
}

REGISTER_OP("GetPSHandleOp")
    .Output("byte_ps_shard: Ref(string)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("use_node_name_sharing: bool = false")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .SetIsStateful()
    .SetShapeFn(TwoElementOutput);

// Register the HashTable op with the currently supported key and value types.
#define REGISTER_KERNEL(key_dtype, value_dtype)                           \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("HashTable")                                                   \
          .Device(DEVICE_CPU)                                             \
          .TypeConstraint<key_dtype>("key_dtype")                         \
          .TypeConstraint<value_dtype>("value_dtype"),                    \
      LookupTableOp<lookup::HashTable<key_dtype, value_dtype>, key_dtype, \
                    value_dtype>)                                         \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("HashTableV2")                                                 \
          .Device(DEVICE_CPU)                                             \
          .TypeConstraint<key_dtype>("key_dtype")                         \
          .TypeConstraint<value_dtype>("value_dtype"),                    \
      LookupTableOp<lookup::HashTable<key_dtype, value_dtype>, key_dtype, \
                    value_dtype>)

REGISTER_KERNEL(int32, double);
REGISTER_KERNEL(int32, float);
REGISTER_KERNEL(int32, int32);
REGISTER_KERNEL(int32, tstring);
REGISTER_KERNEL(int64, double);
REGISTER_KERNEL(int64, float);
REGISTER_KERNEL(int64, int32);
REGISTER_KERNEL(int64, int64);
REGISTER_KERNEL(int64, tstring);
REGISTER_KERNEL(tstring, bool);
REGISTER_KERNEL(tstring, double);
REGISTER_KERNEL(tstring, float);
REGISTER_KERNEL(tstring, int32);
REGISTER_KERNEL(tstring, int64);
REGISTER_KERNEL(tstring, tstring);

#undef REGISTER_KERNEL


REGISTER_OP("PSPullOp")
    .Input("byte_ps_shard: resource")
    .Input("keys: Tin")
    .Input("default_value: Tout")
    .Output("values: Tout")
    .Attr("Tin: type")
    .Attr("Tout: type")
    .SetShapeFn([](InferenceContext *c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle));

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

REGISTER_OP("PSPushOp")
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

REGISTER_OP("PSLoadOp")
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

REGISTER_OP("PSSaveOp")
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
