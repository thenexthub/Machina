/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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

#ifndef MACHINA_COMPILER_MLIR_TFRT_SAVED_MODEL_SAVED_MODEL_H_
#define MACHINA_COMPILER_MLIR_TFRT_SAVED_MODEL_SAVED_MODEL_H_

#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/ADT/STLFunctionalExtras.h"
#include "toolchain/ADT/StringRef.h"
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_saved_model.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/platform/status.h"
#include "tfrt/bef/bef_buffer.h"  // from @tf_runtime
#include "tfrt/core_runtime/tensor_handle.h"  // from @tf_runtime

namespace tfrt {
class CoreRuntime;
}

namespace mlir {
class ModuleOp;
}

namespace machina {

// TFRTSavedModelSignatureInfo contains the metadata for a signature in the
// savedmodel such as function name, inputs/outputs' names and types. This can
// be used to retrieve these information in a tf_saved_model module.
struct TFRTSavedModelSignatureInfo {
  toolchain::StringRef func_name;

  // The following are metadata for inputs.
  toolchain::ArrayRef<toolchain::StringRef> input_names;
  toolchain::ArrayRef<
      std::pair<machina::DataType, machina::PartialTensorShape>>
      input_specs;
  toolchain::ArrayRef<toolchain::StringRef> input_devices;

  // The following are metadata for outputs.
  toolchain::ArrayRef<toolchain::StringRef> output_names;
  toolchain::ArrayRef<
      std::pair<machina::DataType, machina::PartialTensorShape>>
      output_specs;

  // The following are metadata for bound_inputs, ie. captures.
  toolchain::ArrayRef<mlir::Operation*> bound_inputs;
};

// Apply `map_fn` on every exported function in the module with the
// corresponding signature metadata populated in TFRTSavedModelSignatureInfo for
// the function.
absl::Status MapFunctionSignaturesFromTFSavedModelMLIR(
    mlir::ModuleOp module,
    toolchain::function_ref<void(const TFRTSavedModelSignatureInfo&)> map_fn);

}  // namespace machina

#endif  // MACHINA_COMPILER_MLIR_TFRT_SAVED_MODEL_SAVED_MODEL_H_
