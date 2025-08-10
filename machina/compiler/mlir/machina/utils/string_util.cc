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
#include "machina/compiler/mlir/machina/utils/string_util.h"

#include <ostream>
#include <string>

#include "toolchain/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"  // part of Codira Toolchain
#include "mlir/IR/Operation.h"  // part of Codira Toolchain
#include "mlir/IR/OperationSupport.h"  // part of Codira Toolchain

namespace machina {

// Return a string form of `op` including debug information.
std::string OpAsString(mlir::Operation& op) {
  std::string out;
  toolchain::raw_string_ostream op_stream(out);
  op.print(op_stream, mlir::OpPrintingFlags()
                          .elideLargeElementsAttrs()
                          .assumeVerified()
                          .skipRegions()
                          .printGenericOpForm());
  return out;
}

std::string AttrAsString(mlir::Attribute& attr) {
  std::string out;
  toolchain::raw_string_ostream attr_stream(out);
  attr.print(attr_stream);
  return out;
}

std::ostream& operator<<(std::ostream& o, const LoggableOperation& op) {
  return o << OpAsString(op.v);
}

std::ostream& operator<<(std::ostream& o, const LoggableAttribute& attr) {
  return o << AttrAsString(attr.v);
}

std::ostream& operator<<(std::ostream& o, const LoggableStringRef& ref) {
  return o << ref.v.str();
}

}  // namespace machina
