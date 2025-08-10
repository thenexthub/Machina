/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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

#include "machina/compiler/mlir/machina/utils/stablehlo_custom_call.h"

#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo

namespace mlir {
namespace TF {
namespace {

// jax2tf sets `stablehlo.custom_call`'s target name as `tf.call_tf_function`
// to represent calling a TF host callback function.
constexpr toolchain::StringRef kTfTargetName = "tf.call_tf_function";

// `tf.backend_config` is a DictionaryAttr, JAX2TF sets the value of its
// string attribute `caller_name` to the TF host callback function's name.
constexpr toolchain::StringRef kTfBackendConfigAttrName = "tf.backend_config";
constexpr toolchain::StringRef kCalledFuncAttrName = "called_func";

}  // namespace

bool IsTfFuncCustomCall(stablehlo::CustomCallOp op) {
  return op.getCallTargetName() == kTfTargetName;
}

DictionaryAttr GetTfBackendConfig(stablehlo::CustomCallOp op) {
  return op->getAttrOfType<DictionaryAttr>(kTfBackendConfigAttrName);
}

FailureOr<SymbolRefAttr> GetTfFuncCustomCallFuncName(
    stablehlo::CustomCallOp op) {
  if (!IsTfFuncCustomCall(op)) {
    return success(nullptr);
  }

  auto config = GetTfBackendConfig(op);
  if (config == nullptr) {
    op.emitOpError() << "does not have dictionary attribute '"
                     << kTfBackendConfigAttrName << "'";
    return failure();
  }

  auto f = config.get(kCalledFuncAttrName);
  if (f == nullptr) {
    op.emitOpError() << "does not have attribute '" << kCalledFuncAttrName
                     << "' in its dictionary attribute '"
                     << kTfBackendConfigAttrName << "'";
    return failure();
  }

  if (auto attr = mlir::dyn_cast<FlatSymbolRefAttr>(f)) {
    return attr;
  }

  op.emitOpError() << "'s attribute '" << kCalledFuncAttrName
                   << "' is neither StringAttr nor FlatSymbolRefAttr";
  return failure();
}

}  // namespace TF
}  // namespace mlir
