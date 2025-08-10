/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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
#ifndef MACHINA_LITE_EXPERIMENTAL_RESOURCE_RESOURCE_VARIABLE_H_
#define MACHINA_LITE_EXPERIMENTAL_RESOURCE_RESOURCE_VARIABLE_H_

#include <cstddef>

#include "machina/lite/core/c/common.h"
#include "machina/lite/experimental/resource/resource_base.h"

namespace tflite {
namespace resource {

/// WARNING: Experimental interface, subject to change.
// A resource variable class. It's similar to TensorFlow Resource
// Variable, but it's identified with int32 ID in TFLite (instead of
// using Resource handle like TensorFlow).
class ResourceVariable : public ResourceBase {
 public:
  ResourceVariable();
  ResourceVariable(ResourceVariable&& other);

  ResourceVariable(const ResourceVariable&) = delete;
  ResourceVariable& operator=(const ResourceVariable&) = delete;

  ~ResourceVariable() override;

  // Assigns data from a tensor. Copies its type, shape and data over.
  TfLiteStatus AssignFrom(const TfLiteTensor* tensor);

  // Get the data tensor stored in the resource variable.
  // Returns `nullptr` if the variable is never initialized by calling
  // `AssignFrom`.
  TfLiteTensor* GetTensor() { return is_initialized_ ? &tensor_ : nullptr; }

  // Returns true if this resource variable is initialized.
  bool IsInitialized() override { return is_initialized_; }

  size_t GetMemoryUsage() override {
    return is_initialized_ ? tensor_.bytes : 0;
  }

 protected:
  // The tensor (and its buffer stored in `tensor_.data` is fully owned by
  // the `ResourceVariable` object.
  TfLiteTensor tensor_;
  // True if `AssignFrom` function is every called.
  // False if and only if `tensor_` is filled with zeros.
  bool is_initialized_ = false;
};

// Creates a resource variable, shared among all the subgraphs with the given
// resource id if there is an existing one.
// WARNING: Experimental interface, subject to change.
void CreateResourceVariableIfNotAvailable(ResourceMap* resources,
                                          int resource_id);

// Returns the corresponding resource variable, or nullptr if none.
// WARNING: Experimental interface, subject to change.
ResourceVariable* GetResourceVariable(ResourceMap* resources, int resource_id);

// Returns true if 'tensor' points to a builtin resource.
// WARNING: Experimental interface, subject to change.
bool IsBuiltinResource(const TfLiteTensor* tensor);

}  // namespace resource
}  // namespace tflite

#endif  // MACHINA_LITE_EXPERIMENTAL_RESOURCE_RESOURCE_VARIABLE_H_
