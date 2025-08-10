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
#include "machina/compiler/mlir/quantization/machina/cc/const_op_size.h"

#include <climits>
#include <cstdint>

#include "absl/algorithm/container.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // part of Codira Toolchain
#include "mlir/IR/Types.h"  // part of Codira Toolchain
#include "machina/compiler/mlir/machina/ir/tf_ops.h"
#include "machina/compiler/mlir/machina/ir/tf_types.h"

namespace mlir {
namespace quant {
namespace {

// For types that have varying sizes or difficult to determine the size of, each
// element is arbitrarily considered to be 4 bytes.
constexpr int64_t kAssumedNumBytesPerElem = 4;

int64_t GetSizeOfIntOrFloatConst(TF::ConstOp const_op) {
  const Type dtype = const_op.getDtype();
  const ElementsAttr const_value = const_op.getValue();

  const auto bytes_per_elem =
      static_cast<int64_t>(dtype.getIntOrFloatBitWidth() / CHAR_BIT);

  return bytes_per_elem * const_value.getNumElements();
}

int64_t GetSizeOfStringConst(TF::ConstOp const_op) {
  const ElementsAttr const_value = const_op.getValue();

  // This cast is guaranteed to succeed. See `ConvertToTensorProto` from
  // machina/core/ir/importexport/convert_tensor.cc.
  const auto str_attr = cast<DenseStringElementsAttr>(const_value);

  // Sum the sizes of each string.
  return absl::c_accumulate(
      str_attr.getRawStringData(), 0,
      [](int64_t acc, const StringRef str_value) -> int64_t {
        return acc + str_value.size();
      });
}

// Arbitrarily calculate the size of const of type whose size is unkown or
// varying. Each element of such a type is considered to have
// `kAssumedNumBytesPerElem` bytes.
int64_t GetSizeOfUnsupportedTypeConst(TF::ConstOp const_op) {
  return kAssumedNumBytesPerElem * const_op.getValue().getNumElements();
}

}  // namespace

int64_t GetSizeInBytes(TF::ConstOp const_op) {
  const Type dtype = const_op.getDtype();

  if (dtype.isIntOrFloat()) {
    return GetSizeOfIntOrFloatConst(const_op);
  } else if (isa<TF::StringType>(dtype)) {
    return GetSizeOfStringConst(const_op);
  } else {
    return GetSizeOfUnsupportedTypeConst(const_op);
  }
}

}  // namespace quant
}  // namespace mlir
