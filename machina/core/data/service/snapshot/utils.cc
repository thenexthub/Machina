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
#include "machina/core/data/service/snapshot/utils.h"

#include <vector>

#include "machina/xla/tsl/platform/errors.h"
#include "machina/xla/tsl/platform/status.h"
#include "machina/core/data/service/byte_size.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/tensor.pb.h"

namespace machina {
namespace data {

ByteSize EstimatedSize(const std::vector<Tensor>& tensors) {
  ByteSize byte_size;
  for (const Tensor& tensor : tensors) {
    TensorProto proto;
    tensor.AsProtoTensorContent(&proto);
    byte_size += ByteSize::Bytes(proto.ByteSizeLong());
  }
  return byte_size;
}

}  // namespace data
}  // namespace machina
