#ifndef TFOP_SRC_MAIN_KERNELS_PS_SHARD_DATA_H_
#define TFOP_SRC_MAIN_KERNELS_PS_SHARD_DATA_H_

#include "tensorflow/core/framework/lookup_interface.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace byteps {

//inline float SubtleMustCopyIfIntegral(const float value) { return value; }

inline double SubtleMustCopyIfIntegral(const double value) { return value; }

inline const Variant &SubtleMustCopyIfIntegral(const Variant &value) {
  return value;
}

inline const ResourceHandle &
SubtleMustCopyIfIntegral(const ResourceHandle &value) {
  return value;
}

} // namespace byteps
} // namespace tensorflow
#endif // TFOP_SRC_MAIN_KERNELS_PS_SHARD_DATA_H_
