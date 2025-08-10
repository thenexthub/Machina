/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "fuzztest/fuzztest.h"
#include "absl/status/status.h"
#include "machina/cc/saved_model/constants.h"
#include "machina/cc/saved_model/loader.h"
#include "machina/cc/saved_model/tag_constants.h"
#include "machina/xla/tsl/platform/env.h"
#include "machina/xla/tsl/platform/status.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/platform/env.h"
#include "machina/core/platform/status.h"
#include "machina/core/protobuf/config.pb.h"
#include "machina/core/protobuf/saved_model.pb.h"
#include "machina/core/public/session_options.h"
#include "machina/security/fuzzing/cc/core/framework/datatype_domains.h"
#include "machina/security/fuzzing/cc/core/framework/tensor_domains.h"
#include "machina/security/fuzzing/cc/core/framework/tensor_shape_domains.h"

namespace machina::fuzzing {
namespace {

// Fuzzer that loads an arbitrary model and performs inference using a fixed
// input.
void FuzzEndToEnd(
    const SavedModel& model,
    const std::vector<std::pair<std::string, machina::Tensor>>& input_dict) {
  SavedModelBundle bundle;
  const SessionOptions session_options;
  const RunOptions run_options;
  const std::string export_dir = "ram://";
  TF_CHECK_OK(tsl::WriteBinaryProto(machina::Env::Default(),
                                    export_dir + kSavedModelFilenamePb, model));

  absl::Status status = LoadSavedModel(session_options, run_options, export_dir,
                                       {kSavedModelTagServe}, &bundle);
  if (!status.ok()) {
    return;
  }

  // Create output placeholder tensors for results
  std::vector<machina::Tensor> outputs;
  std::vector<std::string> output_names = {"fuzz_out:0", "fuzz_out:1"};
  absl::Status status_run =
      bundle.session->Run(input_dict, output_names, {}, &outputs);
}

FUZZ_TEST(End2EndFuzz, FuzzEndToEnd)
    .WithDomains(
        fuzztest::Arbitrary<SavedModel>(),
        fuzztest::VectorOf(fuzztest::PairOf(fuzztest::Arbitrary<std::string>(),
                                            fuzzing::AnyValidNumericTensor(
                                                fuzzing::AnyValidTensorShape(
                                                    /*max_rank=*/3,
                                                    /*dim_lower_bound=*/0,
                                                    /*dim_upper_bound=*/20),
                                                fuzzing::AnyValidDataType())))
            .WithMaxSize(6));

}  // namespace
}  // namespace machina::fuzzing
