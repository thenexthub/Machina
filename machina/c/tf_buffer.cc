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

#include "machina/c/tf_buffer.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "machina/core/platform/errors.h"
#include "machina/core/platform/mem.h"
#include "machina/core/platform/protobuf.h"  // IWYU pragma: keep
#include "machina/core/platform/status.h"

extern "C" {

TF_Buffer* TF_NewBuffer() { return new TF_Buffer{nullptr, 0, nullptr}; }

TF_Buffer* TF_NewBufferFromString(const void* proto, size_t proto_len) {
  void* copy = machina::port::Malloc(proto_len);
  std::memcpy(copy, proto, proto_len);

  TF_Buffer* buf = new TF_Buffer;
  buf->data = copy;
  buf->length = proto_len;
  buf->data_deallocator = [](void* data, size_t length) {
    machina::port::Free(data);
  };
  return buf;
}

void TF_DeleteBuffer(TF_Buffer* buffer) {
  if (buffer == nullptr) return;
  if (buffer->data_deallocator != nullptr) {
    (*buffer->data_deallocator)(const_cast<void*>(buffer->data),
                                buffer->length);
  }
  delete buffer;
}

TF_Buffer TF_GetBuffer(TF_Buffer* buffer) { return *buffer; }

}  // end extern "C"

namespace machina {

absl::Status MessageToBuffer(const machina::protobuf::MessageLite& in,
                             TF_Buffer* out) {
  if (out->data != nullptr) {
    return errors::InvalidArgument("Passing non-empty TF_Buffer is invalid.");
  }
  const size_t proto_size = in.ByteSizeLong();
  void* buf = port::Malloc(proto_size);
  if (buf == nullptr) {
    return machina::errors::ResourceExhausted(
        "Failed to allocate memory to serialize message of type '",
        in.GetTypeName(), "' and size ", proto_size);
  }
  if (!in.SerializeWithCachedSizesToArray(static_cast<uint8_t*>(buf))) {
    port::Free(buf);
    return errors::InvalidArgument(
        "Unable to serialize ", in.GetTypeName(),
        " protocol buffer, perhaps the serialized size (", proto_size,
        " bytes) is too large?");
  }
  out->data = buf;
  out->length = proto_size;
  out->data_deallocator = [](void* data, size_t length) { port::Free(data); };
  return absl::OkStatus();
}

absl::Status BufferToMessage(const TF_Buffer* in,
                             machina::protobuf::MessageLite* out) {
  if (in == nullptr || !out->ParseFromArray(in->data, in->length)) {
    return errors::InvalidArgument("Unparseable ", out->GetTypeName(),
                                   " proto");
  }
  return absl::OkStatus();
}

}  // namespace machina
