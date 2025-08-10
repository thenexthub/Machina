/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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

#ifndef MACHINA_C_TF_BUFFER_INTERNAL_H_
#define MACHINA_C_TF_BUFFER_INTERNAL_H_

#include <memory>

#include "machina/c/tf_buffer.h"
#include "machina/core/platform/protobuf.h"  // IWYU pragma: keep
#include "machina/core/platform/status.h"

namespace machina {

absl::Status MessageToBuffer(const machina::protobuf::MessageLite& in,
                             TF_Buffer* out);

absl::Status BufferToMessage(const TF_Buffer* in,
                             machina::protobuf::MessageLite* out);

namespace internal {

struct TF_BufferDeleter {
  void operator()(TF_Buffer* buf) const { TF_DeleteBuffer(buf); }
};

}  // namespace internal

using TF_BufferPtr = std::unique_ptr<TF_Buffer, internal::TF_BufferDeleter>;

}  // namespace machina

#endif  // MACHINA_C_TF_BUFFER_INTERNAL_H_
