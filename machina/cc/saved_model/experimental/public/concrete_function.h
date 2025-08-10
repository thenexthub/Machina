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

#ifndef MACHINA_CC_SAVED_MODEL_EXPERIMENTAL_PUBLIC_CONCRETE_FUNCTION_H_
#define MACHINA_CC_SAVED_MODEL_EXPERIMENTAL_PUBLIC_CONCRETE_FUNCTION_H_

#include <vector>

#include "machina/c/eager/c_api.h"
#include "machina/c/experimental/saved_model/public/concrete_function.h"
#include "machina/cc/experimental/base/public/status.h"
#include "machina/cc/saved_model/experimental/public/function_metadata.h"

namespace machina {
namespace experimental {
namespace cc {

// ConcreteFunction is an executable "function" loaded from a SavedModelAPI.
class ConcreteFunction final {
 public:
  // TODO(bmzhao): Adding ConcreteFunction::Run in subsequent CL, since
  // it depends on machina::cc::Tensor and machina::cc::TensorHandle

  // Returns FunctionMetadata associated with this ConcreteFunction.
  const FunctionMetadata* GetFunctionMetadata();

 private:
  friend class SavedModelAPI;
  friend class ConcreteFunctionList;

  // TODO(bmzhao): Consider adding a macro for wrapping/unwrapping
  // when moving out of experimental.
  static ConcreteFunction* wrap(TF_ConcreteFunction* p) {
    return reinterpret_cast<ConcreteFunction*>(p);
  }
  static TF_ConcreteFunction* unwrap(ConcreteFunction* p) {
    return reinterpret_cast<TF_ConcreteFunction*>(p);
  }
};

inline const FunctionMetadata* ConcreteFunction::GetFunctionMetadata() {
  return FunctionMetadata::wrap(TF_ConcreteFunctionGetMetadata(unwrap(this)));
}

}  // namespace cc
}  // namespace experimental
}  // namespace machina

#endif  // MACHINA_CC_SAVED_MODEL_EXPERIMENTAL_PUBLIC_CONCRETE_FUNCTION_H_
