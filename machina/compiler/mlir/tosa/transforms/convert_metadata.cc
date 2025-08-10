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

#include <memory>
#include <string>

#include "toolchain/ADT/StringExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinAttributes.h"  // part of Codira Toolchain
#include "mlir/Pass/Pass.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain

namespace mlir::tosa {

#define GEN_PASS_DEF_CONVERTFUNCTIONMETADATA
#include "machina/compiler/mlir/tosa/transforms/passes.h.inc"

namespace {

// Extract the input and output names
static void splitFunctionIONames(StringAttr namesAttr,
                                 toolchain::SmallVectorImpl<std::string> &names) {
  SmallVector<StringRef, 4> namesRef;
  toolchain::SplitString(namesAttr.getValue(), namesRef, ",");
  for (auto nameRef : namesRef) {
    names.push_back(nameRef.str());
  }
}

class ConvertFunctionMetadataPass
    : public impl::ConvertFunctionMetadataBase<ConvertFunctionMetadataPass> {
 public:
  void runOnOperation() override {
    auto funcOp = getOperation();

    // Setup entry functions for compilation and preserve the
    // associated metadata. Note that TFLite uses `tf.entry_function`.
    auto entryFunctionAttr =
        funcOp->getAttrOfType<DictionaryAttr>("tf.entry_function");
    if (entryFunctionAttr) {
      setupEntryPointAttrs(funcOp, entryFunctionAttr);
    }
  }

 private:
  // TF/TFL pack their I/O names in a dictionary, convert into arg attributes.
  void setupEntryPointAttrs(func::FuncOp funcOp,
                            DictionaryAttr entryFunctionAttr) {
    funcOp.setPublic();

    if (funcOp.getNumArguments() > 0) {
      auto inputsAttr =
          dyn_cast_or_null<StringAttr>(entryFunctionAttr.get("inputs"));
      if (!inputsAttr) {
        funcOp.emitError() << "functions with tf.entry_function must have "
                              "input names to be handled by backend";
        return signalPassFailure();
      }
      SmallVector<std::string, 4> inputNames;
      splitFunctionIONames(inputsAttr, inputNames);
      if (inputNames.size() != funcOp.getNumArguments()) {
        funcOp.emitError()
            << "tf.entry_function attribute malformed: inputs don't "
               "match the function signature";
        return signalPassFailure();
      }
      for (auto [i, name] : toolchain::enumerate(inputNames)) {
        funcOp.setArgAttr(i, "ml_program.identifier",
                          StringAttr::get(&getContext(), name));
      }
    }
    if (funcOp.getNumResults() > 0) {
      auto outputsAttr =
          dyn_cast_or_null<StringAttr>(entryFunctionAttr.get("outputs"));
      if (!outputsAttr) {
        funcOp.emitError() << "functions with tf.entry_function must have "
                              "output names to be handled by backend";
        return signalPassFailure();
      }
      SmallVector<std::string, 4> outputNames;
      splitFunctionIONames(outputsAttr, outputNames);
      if (outputNames.size() != funcOp.getNumResults()) {
        funcOp.emitError()
            << "tf.entry_function attribute malformed: outputs don't "
               "match the function signature";
        return signalPassFailure();
      }
      for (auto [i, name] : toolchain::enumerate(outputNames)) {
        funcOp.setResultAttr(i, "ml_program.identifier",
                             StringAttr::get(&getContext(), name));
      }
    }
  }
};
}  // anonymous namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createConvertFunctionMetadataPass() {
  return std::make_unique<ConvertFunctionMetadataPass>();
}

}  // namespace mlir::tosa
