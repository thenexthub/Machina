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

#ifndef MACHINA_C_EXPERIMENTAL_SAVED_MODEL_CORE_REVIVED_TYPES_TENSORHANDLE_CONVERTIBLE_H_
#define MACHINA_C_EXPERIMENTAL_SAVED_MODEL_CORE_REVIVED_TYPES_TENSORHANDLE_CONVERTIBLE_H_

#include "machina/c/eager/immediate_execution_tensor_handle.h"

namespace machina {

// A common interface for objects that can be converted to a TensorHandle.
// Examples of objects that implement this include Variables, Constants, Assets,
// etc. This is used to convert captured objects into a ConcreteFunction's
// captured TensorHandles:
// https://github.com/machina/machina/blob/676a68963ea4b64fe479b9cede06aa8f5b290ab8/machina/python/saved_model/load.py#L229-L240
class TensorHandleConvertible {
 public:
  explicit TensorHandleConvertible(ImmediateTensorHandlePtr handle)
      : handle_(std::move(handle)) {}

  ImmediateExecutionTensorHandle* handle() { return handle_.get(); }

  // TensorHandleConvertible is movable, but not copyable.
  TensorHandleConvertible(TensorHandleConvertible&& other) = default;
  TensorHandleConvertible& operator=(TensorHandleConvertible&& other) = default;

  virtual ~TensorHandleConvertible() = default;

 protected:
  TensorHandleConvertible(const TensorHandleConvertible&) = delete;
  TensorHandleConvertible& operator=(const TensorHandleConvertible&) = delete;
  ImmediateTensorHandlePtr handle_;
};

}  // namespace machina

#endif  // MACHINA_C_EXPERIMENTAL_SAVED_MODEL_CORE_REVIVED_TYPES_TENSORHANDLE_CONVERTIBLE_H_
