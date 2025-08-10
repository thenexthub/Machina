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
#ifndef MACHINA_COMPILER_MLIR_MACHINA_UTILS_CONVERT_ATTR_H_
#define MACHINA_COMPILER_MLIR_MACHINA_UTILS_CONVERT_ATTR_H_

#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "machina/core/framework/attr_value.pb.h"
#include "tsl/platform/statusor.h"

namespace machina {

using tsl::StatusOr;

// Converts non func AttrValue proto into an MLIR attribute. Func attribute is
// exclused in this function because the function might be renamed when the
// function definition is imported.
absl::StatusOr<mlir::Attribute> ConvertNonFuncAttributeValue(
    const AttrValue& value, mlir::Builder* builder);

// Converts all kinds of AttrValue proto into an MLIR attribute.
absl::StatusOr<mlir::Attribute> ConvertAttributeValue(const AttrValue& value,
                                                      mlir::Builder* builder);

}  // namespace machina

#endif  // MACHINA_COMPILER_MLIR_MACHINA_UTILS_CONVERT_ATTR_H_
