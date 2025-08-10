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
#include "machina/compiler/mlir/quantization/common/func.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/SymbolTable.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "machina/cc/saved_model/signature_constants.h"
#include "machina/compiler/mlir/machina/translate/import_model.h"

namespace mlir::quant {
namespace {

using ::machina::kDefaultServingSignatureDefKey;
using ::machina::kImportModelDefaultGraphFuncName;

// Returns true iff the function's symbol is public.
bool IsPublicFuncOp(func::FuncOp func_op) {
  return SymbolTable::getSymbolVisibility(&*func_op) ==
         SymbolTable::Visibility::Public;
}

}  // namespace

func::FuncOp FindMainFuncOp(ModuleOp module_op) {
  if (const auto main_func_op = module_op.lookupSymbol<func::FuncOp>(
          kImportModelDefaultGraphFuncName);
      main_func_op != nullptr && IsPublicFuncOp(main_func_op)) {
    return main_func_op;
  }

  if (const auto serving_default_func_op =
          module_op.lookupSymbol<func::FuncOp>(kDefaultServingSignatureDefKey);
      serving_default_func_op != nullptr &&
      IsPublicFuncOp(serving_default_func_op)) {
    return serving_default_func_op;
  }

  return nullptr;
}

}  // namespace mlir::quant
