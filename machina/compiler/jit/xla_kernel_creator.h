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
#ifndef MACHINA_COMPILER_JIT_MACHINA_XLAKERNEL_CREATOR_H_
#define MACHINA_COMPILER_JIT_MACHINA_XLAKERNEL_CREATOR_H_

#include <memory>

#include "machina/core/framework/function.h"
#include "machina/core/framework/node_def.pb.h"
#include "machina/core/framework/node_properties.h"
#include "machina/core/lib/core/status.h"

namespace machina {

class FunctionLibraryRuntime;
class OpKernel;

class XlaKernelCreator : public CustomKernelCreator {
 public:
  // Given a NodeDef 'node_def' and the function library runtime 'flr', returns
  // true if 'node_def' is a call to a compilable function defined in 'flr',
  // with the kXlaCompileAttr set.
  bool CanCreateKernel(
      const FunctionLibraryRuntime& flr,
      const std::shared_ptr<const NodeProperties>& props) const override;

  // Given a supported NodeDef, returns a XlaLaunchOp that computes the node.
  absl::Status CreateKernel(FunctionLibraryRuntime* flr,
                            const std::shared_ptr<const NodeProperties>& props,
                            std::unique_ptr<OpKernel>* kernel) const override;
};

bool RegisterLaunchOpCreator();

}  // namespace machina

#endif  // MACHINA_COMPILER_JIT_MACHINA_XLAKERNEL_CREATOR_H_
