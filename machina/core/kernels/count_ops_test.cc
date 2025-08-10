/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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

#include "machina/core/framework/fake_input.h"
#include "machina/core/framework/node_def_builder.h"
#include "machina/core/framework/shape_inference.h"
#include "machina/core/framework/shape_inference_testutil.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/tensor_testutil.h"
#include "machina/core/kernels/ops_testutil.h"
#include "machina/core/lib/core/status_test_util.h"
#include "machina/core/platform/test.h"

namespace machina {
namespace {

TEST_F(OpsTestBase, DenseCountSparseOutputShapeFn) {
  ShapeInferenceTestOp op("DenseCountSparseOutput");
  INFER_OK(op, "[?];?", "[?,1];[?];[1]");
  INFER_OK(op, "[?,?];?", "[?,2];[?];[2]");
}

TEST_F(OpsTestBase, SparseCountSparseOutputShapeFn) {
  ShapeInferenceTestOp op("SparseCountSparseOutput");
  INFER_OK(op, "[?,1];?;?;?", "[?,d0_1];[?];[d0_1]");
  INFER_OK(op, "[?,2];?;?;?", "[?,d0_1];[?];[d0_1]");
}

TEST_F(OpsTestBase, RaggedCountSparseOutputShapeFn) {
  ShapeInferenceTestOp op("RaggedCountSparseOutput");
  INFER_OK(op, "?;[?];?", "[?,2];[?];[2]");
}
}  // namespace
}  // namespace machina
