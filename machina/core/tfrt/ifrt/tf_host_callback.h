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

#ifndef MACHINA_CORE_TFRT_IFRT_TF_HOST_CALLBACK_H_
#define MACHINA_CORE_TFRT_IFRT_TF_HOST_CALLBACK_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "machina/compiler/mlir/tfrt/transforms/ifrt/ifrt_types.h"
#include "machina/core/common_runtime/device_mgr.h"
#include "machina/core/common_runtime/eager/context.h"
#include "machina/core/framework/function.pb.h"
#include "machina/core/protobuf/config.pb.h"

namespace machina {
namespace ifrt_serving {

// A host callback implementation to run a TF graph.
// TODO(b/332774825): Use TFRT executor for host callback.
class TfHostCallback {
 public:
  // Creates a TfHostCallback instance. `device_mgr` ptr is guaranteed to be
  // alive throughout the lifetime of model.
  static absl::StatusOr<std::unique_ptr<TfHostCallback>> Create(
      absl::Span<const machina::FunctionDef> functions,
      absl::string_view entry_function_name,
      absl::Span<const DtypeAndShape> operand_type_and_shapes,
      absl::Span<const DtypeAndShape> result_type_and_shapes,
      machina::DeviceMgr* device_mgr);

  // The host callback function takes two pointer arrays, each element of which
  // points to allocated host buffer in host layout according to corresponding
  // operand or result's shape. The buffers are only guaranteed to be alive
  // during the call.
  absl::Status Call(void** inputs, void** outputs);

 private:
  TfHostCallback(absl::string_view entry_function_name,
                 absl::Span<const DtypeAndShape> operand_type_and_shapes,
                 absl::Span<const DtypeAndShape> result_type_and_shape,
                 machina::EagerContextPtr ctx)
      : ctx_(std::move(ctx)),
        entry_function_name_(entry_function_name),
        operand_type_and_shapes_(operand_type_and_shapes.begin(),
                                 operand_type_and_shapes.end()),
        result_type_and_shapes_(result_type_and_shape.begin(),
                                result_type_and_shape.end()) {}

  // Per-callback TF Eager context.
  machina::EagerContextPtr ctx_;

  // Entry function name to be called on invocation.
  std::string entry_function_name_;

  std::vector<DtypeAndShape> operand_type_and_shapes_;
  std::vector<DtypeAndShape> result_type_and_shapes_;
};

absl::StatusOr<std::unique_ptr<machina::DynamicDeviceMgr>>
CreateTfDynamicDeviceMgr();

}  // namespace ifrt_serving
}  // namespace machina

#endif  // MACHINA_CORE_TFRT_IFRT_TF_HOST_CALLBACK_H_
