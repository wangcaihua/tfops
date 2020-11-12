//
// Created by bytedance on 2020/11/11.
//

#ifndef TFOP_SRC_MAIN_KERNELS_PS_SHARD_DATA_H_
#define TFOP_SRC_MAIN_KERNELS_PS_SHARD_DATA_H_

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/lookup_interface.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {
namespace byteps {

class PSShard : public lookup::LookupInterface {
public:
  TensorShape key_shape() const final { return TensorShape(); }

  TensorShape value_shape() const override { return TensorShape(); }

  Status CheckKeyAndValueTensorsForInsert(const Tensor &keys,
                                          const Tensor &values) override {
    return CheckKeyAndValueTensorsHelper(keys, values);
  }

  Status CheckKeyAndValueTensorsForImport(const Tensor &keys,
                                          const Tensor &values) override {
    return CheckKeyAndValueTensorsHelper(keys, values);
  }

  Status CheckKeyTensorForRemove(const Tensor &keys) override {
    if (keys.dtype() != key_dtype()) {
      return errors::InvalidArgument("Key must be type ", key_dtype(),
                                     " but got ", keys.dtype());
    }
    return CheckKeyShape(keys.shape());
  }

private:
  // virtual ~PSShard() = default;
  Status CheckKeyAndValueTensorsHelper(const Tensor &keys,
                                       const Tensor &values) {
    TF_RETURN_IF_ERROR(CheckKeyAndValueTypes(keys, values));
    TF_RETURN_IF_ERROR(CheckKeyShape(keys.shape()));

    TensorShape expected_value_shape = keys.shape();
    for (int i = 0; i < key_shape().dims(); ++i) {
      expected_value_shape.RemoveDim(expected_value_shape.dims() - 1);
    }
    expected_value_shape.AppendShape(value_shape());
    if (values.shape() != expected_value_shape) {
      return errors::InvalidArgument(
          "Expected shape ", expected_value_shape.DebugString(),
          " for value, got ", values.shape().DebugString());
    }
    return Status::OK();
  }
};

template <class K, class V> class PSShardOfScalars final : public PSShard {
public:
  PSShardOfScalars(OpKernelContext *ctx, OpKernel *kernel) {}

  size_t size() const override {
    tf_shared_lock l(mu_);
    return table_.size();
  }

  Status Find(OpKernelContext *ctx, const Tensor &key, Tensor *value,
              const Tensor &default_value) override;

  Status DoInsert(bool clear, const Tensor &keys, const Tensor &values);

  Status Insert(OpKernelContext *ctx, const Tensor &keys,
                const Tensor &values) override {
    return DoInsert(false, keys, values);
  }

  Status Remove(OpKernelContext *ctx, const Tensor &keys) override {
    return Status::OK();
  }

  Status ImportValues(OpKernelContext *ctx, const Tensor &keys,
                      const Tensor &values) override {
    return DoInsert(true, keys, values);
  }

  Status ExportValues(OpKernelContext *ctx) override;

  int64 MemoryUsed() const override {
    int64 ret = 0;
    tf_shared_lock l(mu_);
    for (unsigned i = 0; i < table_.bucket_count(); ++i) {
      size_t bucket_size = table_.bucket_size(i);
      if (bucket_size == 0) {
        ret++;
      } else {
        ret += bucket_size;
      }
    }
    return sizeof(PSShardOfScalars) + ret;
  }

  DataType key_dtype() const override { return DataTypeToEnum<K>::v(); }

  DataType value_dtype() const override { return DataTypeToEnum<V>::v(); }

private:
  // virtual ~PSShardOfScalars() = default;
  mutable mutex mu_;
  std::unordered_map<K, V> table_ GUARDED_BY(mu_);
};

} // namespace byteps
} // namespace tensorflow

#endif // TFOP_SRC_MAIN_KERNELS_PS_SHARD_DATA_H_
