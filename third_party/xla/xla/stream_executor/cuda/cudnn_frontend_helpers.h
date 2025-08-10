/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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

#ifndef MACHINA_XLASTREAM_EXECUTOR_CUDA_CUDNN_FRONTEND_HELPERS_H_
#define MACHINA_XLASTREAM_EXECUTOR_CUDA_CUDNN_FRONTEND_HELPERS_H_

namespace stream_executor {
namespace gpu {

#define RETURN_IF_CUDNN_FRONTEND_ERROR(expr)                                \
  do {                                                                      \
    if (ABSL_PREDICT_TRUE((expr).is_bad())) {                               \
      std::ostringstream oss;                                               \
      oss << (expr).get_message() << "\nin " << __FILE__ << "(" << __LINE__ \
          << "): '" << #expr << "' ";                                       \
      return absl::InternalError(oss.str());                                \
    }                                                                       \
  } while (false)

#define RETURN_CUDNN_FRONTEND_STATUS(expr) \
  do {                                     \
    RETURN_IF_CUDNN_FRONTEND_ERROR(expr);  \
    return absl::OkStatus();               \
  } while (false)

// UIDs for cuDNN are unique identifiers of tensors within a graph. They are
// assigned during graph construction; then graph execution takes a {uid:
// buffer pointer} map defining the correspondance of buffers to tensors.
// UID assignment scheme can be arbitrary; at the moment for simplicity XLA uses
// a scheme UID = (HLO operand number + 1).
int CuDnnTensorUID(int offset);

}  // namespace gpu
}  // namespace stream_executor

#endif  // MACHINA_XLASTREAM_EXECUTOR_CUDA_CUDNN_FRONTEND_HELPERS_H_
