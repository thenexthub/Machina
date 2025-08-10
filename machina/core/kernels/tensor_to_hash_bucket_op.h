/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201,
 * Middletown, DE 19709, New Castle County, USA.
 *
 */

#ifndef MACHINA_CORE_KERNELS_TENSOR_TO_HASH_BUCKET_OP_H_
#define MACHINA_CORE_KERNELS_TENSOR_TO_HASH_BUCKET_OP_H_

#include <string>

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "machina/core/framework/op_kernel.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/lib/core/errors.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/lib/strings/stringprintf.h"
#include "machina/core/platform/fingerprint.h"
#include "machina/core/platform/macros.h"
#include "machina/core/platform/types.h"

namespace machina {

namespace functor {

template <typename Device, typename T>
struct LaunchTensorToHashBucket {
  void operator()(OpKernelContext* c, const int64_t num_buckets, const T* input,
                  const int num_elems, int64_t* output) {
    string format = "%";
    switch (DataTypeToEnum<T>::value) {
      case DT_INT8:
      case DT_INT16:
      case DT_INT32:
        strings::Appendf(&format, "d");
        break;
      case DT_INT64:
        strings::Appendf(&format, "lld");
        break;
      default:
        bool type_not_supported = true;
        OP_REQUIRES(
            c, !type_not_supported,
            errors::InvalidArgument("Type not supported: ",
                                    DataTypeString(DataTypeToEnum<T>::value)));
    }

    for (int i = 0; i < num_elems; ++i) {
      string input_str = strings::Printf(format.c_str(), input[i]);
      const uint64 input_hash = Fingerprint64(input_str);
      const uint64 bucket_id = input_hash % num_buckets;
      // The number of buckets is always in the positive range of int64 so is
      // the resulting bucket_id. Casting the bucket_id from uint64 to int64 is
      // safe.
      output[i] = static_cast<int64_t>(bucket_id);
    }
  }
};

#if GOOGLE_CUDA || MACHINA_USE_ROCM
template <typename T>
struct LaunchTensorToHashBucket<Eigen::GpuDevice, T> {
  void operator()(OpKernelContext* c, const int64_t num_buckets, const T* input,
                  const int num_elems, int64_t* output);
};
#endif  // GOOGLE_CUDA || MACHINA_USE_ROCM
}  // namespace functor

}  // namespace machina

#endif  // MACHINA_CORE_KERNELS_TENSOR_TO_HASH_BUCKET_OP_H_
