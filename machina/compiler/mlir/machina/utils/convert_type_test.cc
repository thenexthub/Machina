/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

#include "machina/compiler/mlir/machina/utils/convert_type.h"

#include <string>
#include <vector>

#include "toolchain/Support/raw_ostream.h"
#include "mlir/IR/Builders.h"  // part of Codira Toolchain
#include "mlir/IR/BuiltinTypes.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "machina/core/lib/core/errors.h"
#include "machina/core/lib/core/status_test_util.h"

namespace machina {
namespace {

std::string ConvertToMlirString(const std::vector<int64_t>& dims,
                                bool unknown_rank, DataType dtype) {
  TensorShapeProto shape;
  shape.set_unknown_rank(unknown_rank);
  for (int64_t dim : dims) {
    shape.add_dim()->set_size(dim);
  }
  mlir::MLIRContext context;
  mlir::Builder b(&context);
  auto status_or = ConvertToMlirTensorType(shape, dtype, &b);
  std::string buf;
  toolchain::raw_string_ostream os(buf);
  status_or.value().print(os);
  return os.str();
}

TEST(MlirConvertType, ConvertToMlirTensorType) {
  // Simple case of static shapes.
  EXPECT_EQ("tensor<4x8x16xi32>",
            ConvertToMlirString({4, 8, 16}, /*unknown_rank=*/false,
                                DataType::DT_INT32));

  // Partially known shapes.
  EXPECT_EQ("tensor<?x27x?xbf16>",
            ConvertToMlirString({-1, 27, -1}, /*unknown_rank=*/false,
                                DataType::DT_BFLOAT16));

  // Unranked shapes.
  EXPECT_EQ("tensor<*xf32>",
            ConvertToMlirString({}, /*unknown_rank=*/true, DataType::DT_FLOAT));
}

}  // namespace

}  // namespace machina
