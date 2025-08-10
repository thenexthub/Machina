/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 26, 2025.
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
#ifndef MACHINA_CORE_FRAMEWORK_SHAPE_INFERENCE_TESTUTIL_H_
#define MACHINA_CORE_FRAMEWORK_SHAPE_INFERENCE_TESTUTIL_H_

#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "machina/core/framework/node_def.pb.h"
#include "machina/core/framework/shape_inference.h"
#include "machina/core/lib/core/stringpiece.h"
#include "machina/core/platform/types.h"
#include "machina/core/public/version.h"

// Contains utilities for writing tests for shape inference functions.

namespace machina {

class Tensor;

struct ShapeInferenceTestOp {
  typedef std::pair<string, DataType> ShapeAndType;
  explicit ShapeInferenceTestOp(absl::string_view name) : name(string(name)) {}
  string name;
  NodeDef node_def;
  std::vector<const Tensor*> input_tensors;
  std::vector<std::vector<ShapeAndType>*>
      input_resource_handle_shapes_and_types;
  int graph_def_version = TF_GRAPH_DEF_VERSION;
};

namespace shape_inference {

class ShapeInferenceTestutil {
 public:
  // Run shape inference for <op.name>, given inputs specified by <ins>
  // and returns an error if the inferred shape does not match expected_outs.
  //
  // <ins> is a semicolon separated list of shapes. Each shape is formatted
  // according to the formatting per
  // shape_inference::InferenceContext::InferenceContext.
  //
  // <expected_outs> is a semicolon separated list of shapes. Each shape is
  // formatted as one of:
  // * ? - an unknown shape, but not matching an input shape
  // * in0|in2|... - output shape must be the same as one of these input shapes.
  // * [1,?,d0_0|d0_1] - output shape is of known rank, with comma-separated
  //      dimension values.
  //      Each dimension value is one of:
  //      * a constant, which means that constant not equal to a specific input
  //      * ?, which means an unknown dim size not equal to a specific input
  //      * d0_0|d1_2, indicating that the dim size must be equal to one of
  //            the given input dimensions; the first number is the input # and
  //            the second is which dimension in that input it corresponds to.
  // <expected_outs> can be "e"; this is used to indicate that shape inference
  // should have failed.
  static absl::Status InferShapes(ShapeInferenceTestOp op, const string& ins,
                                  const string& expected_outs);

 private:
  ShapeInferenceTestutil() = default;

  // Makes a shape out of 'spec'.
  static absl::Status MakeShapeFromString(
      InferenceContext::ShapeManager* manager, const string& spec,
      ShapeHandle* output);
};

}  // namespace shape_inference

#define INFER_OK(op, i, o)                                                    \
  EXPECT_EQ(machina::shape_inference::ShapeInferenceTestutil::InferShapes( \
                op, i, o),                                                    \
            absl::OkStatus())

#define INFER_ERROR(error_substring, op, i)                                    \
  {                                                                            \
    absl::Status status =                                                      \
        (machina::shape_inference::ShapeInferenceTestutil::InferShapes(     \
            op, i, "e"));                                                      \
    std::string error_message = status.ToString();                             \
    EXPECT_NE(status, absl::OkStatus());                                       \
    EXPECT_TRUE(absl::StrContains(error_message, error_substring))             \
        << "Expected to see '" << error_substring << "' in '" << error_message \
        << "'";                                                                \
  }

}  // namespace machina

#endif  // MACHINA_CORE_FRAMEWORK_SHAPE_INFERENCE_TESTUTIL_H_
