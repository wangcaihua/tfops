#include "ps_kernels.h"

namespace tensorflow {

class ShardOpBaseKernel : public OpKernel {
public:
  explicit ShardOpBaseKernel(OpKernelConstruction *ctx)
      : OpKernel(ctx),
        expected_input_0_(ctx->input_type(0) == DT_RESOURCE ? DT_RESOURCE
                                                            : DT_STRING_REF) {}

protected:
  static Status GetPSShard(OpKernelContext *ctx, PSShard **shard) {
    return byteps::GetPSShard("byte_ps_shard", ctx, shard);
  }

  // Input 0 could be a STRING_REF or a RESOURCE
  const DataType expected_input_0_;

  TF_DISALLOW_COPY_AND_ASSIGN(ShardOpBaseKernel);
};

class AsyncShardOpBaseKernel : public AsyncOpKernel {
public:
  explicit AsyncShardOpBaseKernel(OpKernelConstruction *ctx)
      : AsyncOpKernel(ctx),
        expected_input_0_(ctx->input_type(0) == DT_RESOURCE ? DT_RESOURCE
                                                            : DT_STRING_REF) {}

protected:
  static Status GetPSShard(OpKernelContext *ctx, PSShard **shard) {
    return byteps::GetPSShard("byte_ps_shard", ctx, shard);
  }

  // Input 0 could be a STRING_REF or a RESOURCE
  const DataType expected_input_0_;

  TF_DISALLOW_COPY_AND_ASSIGN(AsyncShardOpBaseKernel);
};

class PSAddMetaOp : public ShardOpBaseKernel {
public:
  explicit PSAddMetaOp(OpKernelConstruction *ctx) : ShardOpBaseKernel(ctx) {}

  void Compute(OpKernelContext *ctx) override {
    PSShard *shard;
    OP_REQUIRES_OK(ctx, GetPSShard(ctx, &shard));
    core::ScopedUnref unref_me(shard);

    DataTypeVector expected_inputs = {expected_input_0_, shard->key_dtype()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));

    const Tensor &key = ctx->input(1);
    const Tensor &default_value = ctx->input(2);
    OP_REQUIRES_OK(ctx, shard->CheckFindArguments(key, default_value));

    TensorShape output_shape = key.shape();
    output_shape.RemoveLastDims(shard->key_shape().dims());
    output_shape.AppendShape(shard->value_shape());
    Tensor *out;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("values", output_shape, &out));

    OP_REQUIRES_OK(ctx, shard->Find(ctx, key, out, default_value));
  }
};

REGISTER_KERNEL_BUILDER(Name("PSAddMeta").Device(DEVICE_CPU), PSAddMetaOp);

class PSPullOp : public ShardOpBaseKernel {
public:
  explicit PSPullOp(OpKernelConstruction *ctx) : ShardOpBaseKernel(ctx) {}

  void Compute(OpKernelContext *ctx) override {
    PSShard *shard;
    OP_REQUIRES_OK(ctx, GetPSShard(ctx, &shard));
    core::ScopedUnref unref_me(shard);

    // DataTypeVector expected_inputs = {expected_input_0_, shard->key_dtype(),
    //                                  shard->value_dtype()};
    // OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));

    const Tensor &key = ctx->input(1);
    const Tensor &default_value = ctx->input(2);
    // OP_REQUIRES_OK(ctx, shard->CheckFindArguments(key, default_value));

    TensorShape output_shape = key.shape();
    output_shape.RemoveLastDims(shard->key_shape().dims());
    output_shape.AppendShape(shard->value_shape());
    Tensor *out;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("values", output_shape, &out));

    OP_REQUIRES_OK(ctx, shard->Find(ctx, key, out, default_value));
    std::cout << "PSPullOp: compute " << std::endl;
  }
};

REGISTER_KERNEL_BUILDER(Name("PSPull").Device(DEVICE_CPU), PSPullOp);

class PSPushOp : public ShardOpBaseKernel {
public:
  explicit PSPushOp(OpKernelConstruction *ctx) : ShardOpBaseKernel(ctx) {}

  void Compute(OpKernelContext *ctx) override {
    PSShard *shard;
    OP_REQUIRES_OK(ctx, GetPSShard(ctx, &shard));
    core::ScopedUnref unref_me(shard);

    //    DataTypeVector expected_inputs = {expected_input_0_,
    //    shard->key_dtype(), shard->value_dtype()};
    //    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));

    const Tensor &keys = ctx->input(1);
    const Tensor &values = ctx->input(2);
    // OP_REQUIRES_OK(ctx, shard->CheckKeyAndValueTensorsForInsert(keys, values));

    int64 memory_used_before = 0;
    if (ctx->track_allocations()) {
      memory_used_before = shard->MemoryUsed();
    }

    OP_REQUIRES_OK(ctx, shard->Insert(ctx, keys, values));
    if (ctx->track_allocations()) {
      ctx->record_persistent_memory_allocation(shard->MemoryUsed() -
                                               memory_used_before);
    }

    std::cout << "PSPushOp: compute " << std::endl;
  }
};

REGISTER_KERNEL_BUILDER(Name("PSPush").Device(DEVICE_CPU), PSPushOp);

class PSLoadOp : public ShardOpBaseKernel {
public:
  explicit PSLoadOp(OpKernelConstruction *ctx) : ShardOpBaseKernel(ctx) {}

  void Compute(OpKernelContext *ctx) override {
    PSShard *shard;
    OP_REQUIRES_OK(ctx, GetPSShard(ctx, &shard));
    core::ScopedUnref unref_me(shard);

    DataTypeVector expected_inputs = {expected_input_0_, shard->key_dtype()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));

    const Tensor &keys = ctx->input(1);
    const Tensor &values = ctx->input(2);
    OP_REQUIRES_OK(ctx, shard->CheckKeyAndValueTensorsForImport(keys, values));

    int64 memory_used_before = 0;
    if (ctx->track_allocations()) {
      memory_used_before = shard->MemoryUsed();
    }

    OP_REQUIRES_OK(ctx, shard->ImportValues(ctx, keys, values));
    if (ctx->track_allocations()) {
      ctx->record_persistent_memory_allocation(shard->MemoryUsed() -
                                               memory_used_before);
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("PSLoad").Device(DEVICE_CPU), PSLoadOp);

class PSSaveOp : public ShardOpBaseKernel {
public:
  explicit PSSaveOp(OpKernelConstruction *ctx) : ShardOpBaseKernel(ctx) {}

  void Compute(OpKernelContext *ctx) override {
    PSShard *shard;
    OP_REQUIRES_OK(ctx, GetPSShard(ctx, &shard));
    core::ScopedUnref unref_me(shard);

    DataTypeVector expected_inputs = {expected_input_0_, shard->key_dtype()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));

    OP_REQUIRES_OK(ctx, shard->ExportValues(ctx));
  }
};

REGISTER_KERNEL_BUILDER(Name("PSSave").Device(DEVICE_CPU), PSSaveOp);

// Register the ScalarHashTable op with the currently supported
// key and value types.
#define REGISTER_KERNEL(key_dtype, value_dtype)                                \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("GetPSHandle")                                                      \
          .Device(DEVICE_CPU)                                                  \
          .TypeConstraint<key_dtype>("key_dtype")                              \
          .TypeConstraint<value_dtype>("value_dtype"),                         \
      GetPSHandleOp<byteps::PSShardOfScalars<key_dtype, value_dtype>,          \
                    key_dtype, value_dtype>)

REGISTER_KERNEL(int32, double);
REGISTER_KERNEL(int32, float);
REGISTER_KERNEL(int32, int32);
REGISTER_KERNEL(int64, double);
REGISTER_KERNEL(int64, float);
REGISTER_KERNEL(int64, int32);
REGISTER_KERNEL(int64, int64);

#undef REGISTER_KERNEL
} // namespace tensorflow