/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
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

#include "machina/c/eager/abstract_tensor_handle.h"

namespace machina {

std::string AbstractTensorHandle::DebugString() const {
  PartialTensorShape shape;
  absl::Status s = Shape(&shape);
  std::string shape_string;
  if (!s.ok()) {
    shape_string = "<error computing shape>";
  } else {
    shape_string = shape.DebugString();
  }
  return absl::StrCat("TensorHandle(shape=", shape_string,
                      ", dtype=", DataType_Name(DataType()),
                      ", type=", FullType().DebugString(), ")");
}

absl::Status AbstractTensorHandle::TensorHandleStatus() const {
  // Tensor handles in current runtime don't carry error info and this method
  // should always return OK status.
  return absl::OkStatus();
}

}  // namespace machina
