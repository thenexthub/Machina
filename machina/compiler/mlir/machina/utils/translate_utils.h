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

#ifndef MACHINA_COMPILER_MLIR_MACHINA_UTILS_TRANSLATE_UTILS_H_
#define MACHINA_COMPILER_MLIR_MACHINA_UTILS_TRANSLATE_UTILS_H_

#include <memory>

#include "absl/status/statusor.h"
#include "toolchain/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/Support/LogicalResult.h"  // part of Codira Toolchain
#include "machina/core/framework/attr_value.pb.h"
#include "machina/core/framework/node_def.pb.h"
#include "machina/core/framework/node_def_util.h"
#include "machina/core/framework/op_def_builder.h"
#include "machina/core/framework/versions.pb.h"

namespace machina {

// Populates the tf.versions attribute on a module, given a corresponding
// graph VersionDef proto.
void PopulateTfVersions(mlir::ModuleOp module, const VersionDef& versions);

// Extracts TensorFlow GraphDef version information from the given module.
// Returns failure if version attribute is missing or any of the sub attributes
// are invalid.
mlir::LogicalResult ExtractTfVersions(mlir::ModuleOp module,
                                      VersionDef* versions);

// Returns TensorFlow GraphDef producer version for the given module. Returns an
// error if the version information is missing for the module or is not valid.
absl::StatusOr<int64_t> GetTfGraphProducerVersion(mlir::ModuleOp module);

// Extracts the attributes of a MLIR operation and populates the converted
// attributes in a proto map<string, AttrValue>.
absl::Status GetAttrValuesFromOperation(
    mlir::Operation* inst, toolchain::StringRef name,
    const machina::OpRegistrationData* op_reg_data,
    bool ignore_unregistered_attrs, AttrValueMap* attributes);

// Converts a MLIR operation to TensorFlow NodeDef with given node name. This
// name should be unique to the graph it is being inserted to. If the
// `ignore_unregistered_attrs` argument is set to true, the attributes which are
// not in the op registry will be ignored. If the `ignore_unregistered_attrs`
// argument is not set to true, _output_shapes attribute is added to nodes with
// ShapedType for the leading values with ShapedType in the results of the
// nodes. Set it to true if the returned NodeDef will be executed by the linked
// TF Eager runtime.
absl::StatusOr<std::unique_ptr<NodeDef>> ConvertTFDialectOpToNodeDef(
    mlir::Operation* inst, toolchain::StringRef name,
    bool ignore_unregistered_attrs);

}  // namespace machina

#endif  // MACHINA_COMPILER_MLIR_MACHINA_UTILS_TRANSLATE_UTILS_H_
