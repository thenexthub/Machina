/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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

#ifndef MACHINA_CORE_IR_IMPORTEXPORT_SAVEDMODEL_EXPORT_H_
#define MACHINA_CORE_IR_IMPORTEXPORT_SAVEDMODEL_EXPORT_H_

#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "machina/core/framework/graph_debug_info.pb.h"
#include "machina/core/platform/errors.h"
#include "machina/core/platform/status.h"
#include "machina/core/protobuf/saved_model.pb.h"

namespace mlir {
namespace tfg {

// Given an MLIR module, returns a `output_saved_model` SavedModel.
// The module must contain at most a single Graph operation and zero or more
// TFFunc operations. `original_saved_model` is used as only a GraphDef portion
// of a saved model represented in the MLIR module.
absl::Status ExportMlirToSavedModel(
    mlir::ModuleOp module, const machina::SavedModel &original_saved_model,
    machina::SavedModel *output_saved_model);

}  // namespace tfg
}  // namespace mlir

#endif  // MACHINA_CORE_IR_IMPORTEXPORT_SAVEDMODEL_EXPORT_H_
