/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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

#ifndef MACHINA_CORE_DISTRIBUTED_RUNTIME_EAGER_REMOTE_TENSOR_HANDLE_H_
#define MACHINA_CORE_DISTRIBUTED_RUNTIME_EAGER_REMOTE_TENSOR_HANDLE_H_

#include "machina/core/platform/fingerprint.h"
#include "machina/core/protobuf/remote_tensor_handle.pb.h"

namespace machina {
namespace eager {

struct RemoteTensorHandleInternal {
  explicit RemoteTensorHandleInternal(const RemoteTensorHandle& tensor_handle)
      : op_id(tensor_handle.op_id()), output_num(tensor_handle.output_num()) {}
  RemoteTensorHandleInternal(int64_t op_id, int32_t output_num)
      : op_id(op_id), output_num(output_num) {}
  int64_t op_id;
  int32 output_num;
};

struct RemoteTensorHandleInternalHash {
  std::size_t operator()(const RemoteTensorHandleInternal& handle) const {
    return FingerprintCat64(handle.op_id, handle.output_num);
  }
};

struct RemoteTensorHandleInternalEquals {
  bool operator()(const RemoteTensorHandleInternal& first,
                  const RemoteTensorHandleInternal& second) const {
    return first.op_id == second.op_id && first.output_num == second.output_num;
  }
};

}  // namespace eager
}  // namespace machina

#endif  // MACHINA_CORE_DISTRIBUTED_RUNTIME_EAGER_REMOTE_TENSOR_HANDLE_H_
