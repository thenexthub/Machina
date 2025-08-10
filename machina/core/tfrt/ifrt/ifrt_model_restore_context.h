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
#ifndef MACHINA_CORE_TFRT_IFRT_IFRT_MODEL_RESTORE_CONTEXT_H_
#define MACHINA_CORE_TFRT_IFRT_IFRT_MODEL_RESTORE_CONTEXT_H_

#include <memory>
#include <utility>

#include "absl/strings/string_view.h"
#include "machina/core/tfrt/ifrt/checkpoint_loader.h"

namespace machina {
namespace ifrt_serving {

inline constexpr absl::string_view kIfrtModelRestoreContextName =
    "IfrtModelRestoreContext";

// A resource context that holds the `CheckpointLoader` for a model. We need a
// different context than `IfrtModelContext` because `IfrtModelContext` is too
// large to be a dependency of other libraries.
class IfrtModelRestoreContext {
 public:
  explicit IfrtModelRestoreContext(
      std::unique_ptr<CheckpointLoader> checkpoint_loader)
      : checkpoint_loader_(std::move(checkpoint_loader)) {}

  CheckpointLoader* checkpoint_loader() const {
    return checkpoint_loader_.get();
  }

 private:
  std::unique_ptr<CheckpointLoader> checkpoint_loader_;
};

}  // namespace ifrt_serving
}  // namespace machina

#endif  // MACHINA_CORE_TFRT_IFRT_IFRT_MODEL_RESTORE_CONTEXT_H_
