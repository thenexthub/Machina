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
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "machina/cc/saved_model/loader.h"
#include "machina/core/lib/core/status_test_util.h"
#include "machina/core/platform/resource_loader.h"
#include "machina/core/tfrt/saved_model/saved_model_testutil.h"

namespace machina {
namespace tfrt_stub {
namespace {

TEST(SavedModelTest, BasicError) {
  std::string saved_model_dir = machina::GetDataDependencyFilepath(
      "machina/core/runtime_fallback/test/saved_model/basic_v1");

  TFRTSavedModelTest test(saved_model_dir);

  std::vector<machina::Tensor> inputs;
  inputs.push_back(
      CreateTfTensor<int32_t>(/*shape=*/{1, 3}, /*data=*/{1, 1, 1}));

  std::vector<machina::Tensor> outputs;
  EXPECT_FALSE(
      test.GetSavedModel()->Run({}, "serving_default", inputs, &outputs).ok());
}

}  // namespace
}  // namespace tfrt_stub
}  // namespace machina
