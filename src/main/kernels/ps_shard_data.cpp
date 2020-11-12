#include "ps_utils.h"

namespace tensorflow {
namespace byteps {

Status GetPSShardHandle(StringPiece input_name, OpKernelContext *ctx,
                        string *container, string *shared_handle) {
  {
    mutex *mu;
    TF_RETURN_IF_ERROR(ctx->input_ref_mutex(input_name, &mu));
    mutex_lock l(*mu);
    Tensor tensor;
    TF_RETURN_IF_ERROR(ctx->mutable_input(input_name, &tensor, true));
    if (tensor.NumElements() != 2) {
      return errors::InvalidArgument(
          "Lookup table handle must be scalar, but had shape: ",
          tensor.shape().DebugString());
    }
    auto h = tensor.flat<string>();
    *container = h(0);
    *shared_handle = h(1);
  }
  return Status::OK();
}

Status GetResourcePSShard(StringPiece input_name, OpKernelContext *ctx,
                          PSShard **shard) {
  const Tensor *handle_tensor;
  TF_RETURN_IF_ERROR(ctx->input(input_name, &handle_tensor));
  const ResourceHandle &handle = handle_tensor->scalar<ResourceHandle>()();
  return LookupResource(ctx, handle, shard);
}

Status GetReferencePSShard(StringPiece input_name, OpKernelContext *ctx,
                           PSShard **table) {
  string container;
  string table_handle;
  TF_RETURN_IF_ERROR(
      GetPSShardHandle(input_name, ctx, &container, &table_handle));
  return ctx->resource_manager()->Lookup(container, table_handle, table);
}

Status GetPSShard(StringPiece input_name, OpKernelContext *ctx,
                  PSShard **shard) {
  DataType handle_dtype;
  TF_RETURN_IF_ERROR(ctx->input_dtype(input_name, &handle_dtype));
  if (handle_dtype == DT_RESOURCE) {
    return GetResourcePSShard(input_name, ctx, shard);
  } else {
    return GetReferencePSShard(input_name, ctx, shard);
  }
}

Status CheckShardDataTypes(const PSShard &shard, DataType key_dtype,
                           DataType value_dtype, const string &shard_name) {
  if (shard.key_dtype() != key_dtype || shard.value_dtype() != value_dtype) {
    return errors::InvalidArgument(
        "Conflicting key/value dtypes ", DataTypeString(key_dtype), "->",
        DataTypeString(value_dtype), " with ",
        DataTypeString(shard.key_dtype()), "-",
        DataTypeString(shard.value_dtype()), " for table ", shard_name);
  }
  return Status::OK();
}
} // namespace byteps
} // namespace tensorflow
