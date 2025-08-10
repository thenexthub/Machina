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

#ifdef TFType_DIALECT
#else
#define TFType_DIALECT

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"


// ODS Definition for the dialect, see https://mlir.toolchain.org/docs/OpDefinitions/
// for more information.


//===----------------------------------------------------------------------===//
// TF Type dialect definitions
//===----------------------------------------------------------------------===//

def TFTypeDialect : Dialect {
  let name = "tf_type";

  let summary = "This dialect defines TensorFlow Types.";
  let description = [{
    This dialect is only provide types and attributes used by various dialects
    in TensorFlow, but does not include any operation.
    It is separated from the other dialect for modularity and binary-size
    control.
  }];

  let cppNamespace = "::mlir::tf_type";
  let extraClassDeclaration = [{
     ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const;
     void printType(::mlir::Type type, ::mlir::DialectAsmPrinter &printer) const;
  }];
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// TFType dialect types and attributes definitions
//===----------------------------------------------------------------------===//

include "machina/core/ir/types/types.td"
include "machina/core/ir/types/attributes.td"

#endif // TFG_DIALECT
