#ifndef TFOP_SRC_MAIN_KERNELS_PS_KERNELS_H_
#define TFOP_SRC_MAIN_KERNELS_PS_KERNELS_H_

#define EIGEN_USE_THREADS

#include <string>
#include <type_traits>
#include <utility>

#include "ps_shard_data.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/core/refcount.h"

namespace tensorflow {

using byteps::CheckShardDataTypes;
using byteps::PSShard;

template <class Container, class key_dtype, class value_dtype>
class GetPSHandleOp : public OpKernel {
public:
  // ctx is not owned by this class.
  explicit GetPSHandleOp(OpKernelConstruction *ctx)
      : OpKernel(ctx), is_shard_handle_set_(false) {
    if (ctx->output_type(0) == DT_RESOURCE) {
      OP_REQUIRES_OK(ctx, ctx->allocate_persistent(tensorflow::DT_RESOURCE,
                                                   tensorflow::TensorShape({}),
                                                   &shard_handle_, nullptr));
    } else {
      OP_REQUIRES_OK(ctx, ctx->allocate_persistent(tensorflow::DT_STRING,
                                                   tensorflow::TensorShape({2}),
                                                   &shard_handle_, nullptr));
    }
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("use_node_name_sharing", &use_node_name_sharing_));
    std::cout << "GetPSHandleOp: create instance" << std::endl;
  }

  // ctx is not owned by this function.
  void Compute(OpKernelContext *ctx) override {
    mutex_lock l(mu_);

    if (!is_shard_handle_set_) {
      OP_REQUIRES_OK(ctx, cinfo_.Init(ctx->resource_manager(), def(),
                                      use_node_name_sharing_));
    }

    auto creator = [ctx, this](PSShard **ret) {
      PSShard *container = new Container(ctx, this);
      if (!ctx->status().ok()) {
        container->Unref();
        return ctx->status();
      }
      if (ctx->track_allocations()) {
        ctx->record_persistent_memory_allocation(
            container->MemoryUsed() + shard_handle_.AllocatedBytes());
      }
      *ret = container;
      std::cout << "GetPSHandleOp: create data" << std::endl;
      return Status::OK();
    };

    PSShard *shard = nullptr;
    OP_REQUIRES_OK(ctx,
                   cinfo_.resource_manager()->template LookupOrCreate<PSShard>(
                       cinfo_.container(), cinfo_.name(), &shard, creator));
    core::ScopedUnref unref_me(shard);

    OP_REQUIRES_OK(ctx, byteps::CheckShardDataTypes(
                            *shard, DataTypeToEnum<key_dtype>::v(),
                            DataTypeToEnum<value_dtype>::v(), cinfo_.name()));

    if (ctx->expected_output_dtype(0) == DT_RESOURCE) {
      if (!is_shard_handle_set_) {
        auto h =
            shard_handle_.AccessTensor(ctx)->template scalar<ResourceHandle>();
        h() =
            MakeResourceHandle<PSShard>(ctx, cinfo_.container(), cinfo_.name());
      }
      ctx->set_output(0, *shard_handle_.AccessTensor(ctx));
    } else {
      if (!is_shard_handle_set_) {
        auto h = shard_handle_.AccessTensor(ctx)->template flat<string>();
        h(0) = cinfo_.container();
        h(1) = cinfo_.name();
      }
      ctx->set_output_ref(0, &mu_, shard_handle_.AccessTensor(ctx));
    }
    std::cout << "GetPSHandleOp: compute" << std::endl;
    is_shard_handle_set_ = true;
  }

  ~GetPSHandleOp() override {
    // If the table object was not shared, delete it.
    if (is_shard_handle_set_ && cinfo_.resource_is_private_to_kernel()) {
      if (!cinfo_.resource_manager()
               ->template Delete<PSShard>(cinfo_.container(), cinfo_.name())
               .ok()) {
        // Do nothing; the resource can have been deleted by session resets.
      }
    }

    std::cout << "GetPSHandleOp: destroy" << std::endl;
  }

private:
  mutex mu_;
  PersistentTensor shard_handle_ GUARDED_BY(mu_);
  bool is_shard_handle_set_;
  ContainerInfo cinfo_;
  bool use_node_name_sharing_;

  TF_DISALLOW_COPY_AND_ASSIGN(GetPSHandleOp);
};

} // namespace tensorflow
#endif // TFOP_SRC_MAIN_KERNELS_PS_KERNELS_H_
