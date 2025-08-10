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

#ifndef MACHINA_COMPILER_TF2MACHINA_MACHINA_XLA_RESOURCE_OPERATION_TABLE_H_
#define MACHINA_COMPILER_TF2MACHINA_MACHINA_XLA_RESOURCE_OPERATION_TABLE_H_

#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "machina/core/platform/logging.h"

// Exposes information about the resource operations supported by tf2xla in a
// structured form.

namespace machina {
enum class XlaResourceOpKind {
  kRead,      // Only reads from resources.
  kWrite,     // Only writes to resources.
  kReadWrite  // Reads from and writes to resources.
};

enum class XlaResourceKind {
  kVariable,    // Operates on resource variables.
  kStack,       // Operates on stacks.
  kTensorArray  // Operates on tensor arrays.
};

class XlaResourceOpInfo {
 public:
  explicit XlaResourceOpInfo(XlaResourceOpKind op_kind,
                             XlaResourceKind resource_kind)
      : op_kind_(op_kind), resource_kind_(resource_kind) {}

  XlaResourceOpKind kind() const { return op_kind_; }
  XlaResourceKind resource_kind() const { return resource_kind_; }

  static absl::string_view XlaResourceOpKindToString(XlaResourceOpKind op_kind);

 private:
  XlaResourceOpKind op_kind_;
  XlaResourceKind resource_kind_;
};

// Returns a XlaResourceOpInfo describing `op` if it is a resource operation
// supported by tf2xla, otherwise returns null (i.e. if this returns null then
// `op` is either not a resource operation or is unsupported by XLA).
const XlaResourceOpInfo* GetResourceOpInfoForOp(absl::string_view op);

namespace resource_op_table_internal {
// NB! Implementation detail exposed for unit testing, do not use.
//
// Returns the set of resource operations known by this module.
std::vector<absl::string_view> GetKnownResourceOps();
}  // namespace resource_op_table_internal

}  // namespace machina

#endif  // MACHINA_COMPILER_TF2MACHINA_MACHINA_XLA_RESOURCE_OPERATION_TABLE_H_
