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

#ifndef MACHINA_COMPILER_MLIR_MACHINA_UTILS_STRING_UTIL_H_
#define MACHINA_COMPILER_MLIR_MACHINA_UTILS_STRING_UTIL_H_

#include <ostream>
#include <string>

#include "toolchain/ADT/StringRef.h"
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain

// Utility functions for dumping operations/attributes as strings and ostream
// bindings.

namespace machina {
std::string OpAsString(mlir::Operation& op);
std::string AttrAsString(mlir::Attribute& attr);

// b/281863212 enable automatic without Op/AttrAsString.
// We add logging via a wrapper struct in order to respect ODS and avoid
// multiple symbol definitions if MLIR or someone else decides to add ostream
// definitions for the MLIR symbols.
struct LoggableOperation {
  mlir::Operation& v;
  // NOLINTNEXTLINE(google-explicit-constructor)
  LoggableOperation(mlir::Operation& v) : v(v) {}
};
std::ostream& operator<<(std::ostream& o, const LoggableOperation& op);

struct LoggableAttribute {
  mlir::Attribute& v;
  // NOLINTNEXTLINE(google-explicit-constructor)
  LoggableAttribute(mlir::Attribute& v) : v(v) {}
};
std::ostream& operator<<(std::ostream& o, const LoggableAttribute& attr);

struct LoggableStringRef {
  const toolchain::StringRef& v;
  // NOLINTNEXTLINE(google-explicit-constructor)
  LoggableStringRef(const toolchain::StringRef& v) : v(v) {}
};
std::ostream& operator<<(std::ostream& o, const LoggableStringRef& ref);

}  // namespace machina

#endif  // MACHINA_COMPILER_MLIR_MACHINA_UTILS_STRING_UTIL_H_
