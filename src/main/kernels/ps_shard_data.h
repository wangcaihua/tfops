#ifndef TFOP_SRC_MAIN_KERNELS_PS_SHARD_DATA_CPP_
#define TFOP_SRC_MAIN_KERNELS_PS_SHARD_DATA_CPP_
#define EIGEN_USE_THREADS

#include <string>
#include <type_traits>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "ps_utils.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/lookup_interface.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/hash/hash.h"
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

Status GetPSShard(StringPiece input_name, OpKernelContext *ctx,
                  PSShard **shard);

// Verify that the given key_dtype and value_dtype matches the corresponding
// table's data types.
Status CheckShardDataTypes(const PSShard &shard, DataType key_dtype,
                           DataType value_dtype, const string &table_name);

template <class K, class V> class PSShardOfScalars final : public PSShard {
public:
  PSShardOfScalars(OpKernelContext *ctx, OpKernel *kernel) {}

  size_t size() const override {
    tf_shared_lock l(mu_);
    return table_.size();
  }

  Status Find(OpKernelContext *ctx, const Tensor &key, Tensor *value,
              const Tensor &default_value) override {
    const auto key_values = key.flat<K>();
    auto value_values = value->flat<V>();
    const auto default_flat = default_value.flat<V>();

    int64 total = value_values.size();
    int64 default_total = default_flat.size();
    bool is_full_size_default = (total == default_total);

    tf_shared_lock l(mu_);
    for (int64 i = 0; i < key_values.size(); ++i) {
      // is_full_size_default is true:
      //   Each key has an independent default value, key_values(i)
      //   corresponding uses default_flat(i) as its default value.
      //
      // is_full_size_default is false:
      //   All keys will share the default_flat(0) as default value.
      value_values(i) = gtl::FindWithDefault(
          table_, SubtleMustCopyIfIntegral(key_values(i)),
          is_full_size_default ? default_flat(i) : default_flat(0));

      auto got = table_.find(key_values(i));
      if (got != table_.end()) {
        std::cout << "find key: " << key_values(i) << "\tvalue: " << got->second
                  << std::endl;
      } else {
        std::cout << "find key: " << key_values(i) << "\tnot found"
                  << std::endl;
      }
    }

    return Status::OK();
  }

  Status DoInsert(bool clear, const Tensor &keys, const Tensor &values) {
    const auto key_values = keys.flat<K>();
    const auto value_values = values.flat<V>();

    mutex_lock l(mu_);
    if (clear) {
      table_.clear();
    }
    for (int64 i = 0; i < key_values.size(); ++i) {
      gtl::InsertOrUpdate(&table_, SubtleMustCopyIfIntegral(key_values(i)),
                          SubtleMustCopyIfIntegral(value_values(i)));
      std::cout << "insert key: " << key_values(i)
                << "\tvalue: " << value_values(i) << std::endl;
    }
    return Status::OK();
  }

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

  Status ExportValues(OpKernelContext *ctx) override {
    tf_shared_lock l(mu_);
    int64 size = table_.size();

    Tensor *keys;
    Tensor *values;
    TF_RETURN_IF_ERROR(
        ctx->allocate_output("keys", TensorShape({size}), &keys));
    TF_RETURN_IF_ERROR(
        ctx->allocate_output("values", TensorShape({size}), &values));

    auto keys_data = keys->flat<K>();
    auto values_data = values->flat<V>();
    int64 i = 0;
    for (auto it = table_.begin(); it != table_.end(); ++it, ++i) {
      keys_data(i) = it->first;
      values_data(i) = it->second;
    }
    return Status::OK();
  }

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
  std::unordered_map<K, V> table_;
};

} // namespace byteps
} // namespace tensorflow

#endif // TFOP_SRC_MAIN_KERNELS_PS_SHARD_DATA_CPP_
