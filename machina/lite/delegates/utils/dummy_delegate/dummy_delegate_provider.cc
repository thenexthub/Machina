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
#include <string>
#include <utility>
#include <vector>

#include "machina/lite/c/c_api_types.h"
#include "machina/lite/delegates/utils/dummy_delegate/dummy_delegate.h"
#include "machina/lite/tools/command_line_flags.h"
#include "machina/lite/tools/delegates/delegate_provider.h"
#include "machina/lite/tools/tool_params.h"

namespace tflite {
namespace tools {

class DummyDelegateProvider : public DelegateProvider {
 public:
  DummyDelegateProvider() {
    default_params_.AddParam("use_dummy_delegate",
                             ToolParam::Create<bool>(false));
  }

  std::vector<Flag> CreateFlags(ToolParams* params) const final;

  void LogParams(const ToolParams& params, bool verbose) const final;

  TfLiteDelegatePtr CreateTfLiteDelegate(const ToolParams& params) const final;
  std::pair<TfLiteDelegatePtr, int> CreateRankedTfLiteDelegate(
      const ToolParams& params) const final;

  std::string GetName() const final { return "DummyDelegate"; }
};
REGISTER_DELEGATE_PROVIDER(DummyDelegateProvider);

std::vector<Flag> DummyDelegateProvider::CreateFlags(ToolParams* params) const {
  std::vector<Flag> flags = {CreateFlag<bool>("use_dummy_delegate", params,
                                              "use the dummy delegate.")};
  return flags;
}

void DummyDelegateProvider::LogParams(const ToolParams& params,
                                      bool verbose) const {
  LOG_TOOL_PARAM(params, bool, "use_dummy_delegate", "Use dummy test delegate",
                 verbose);
}

TfLiteDelegatePtr DummyDelegateProvider::CreateTfLiteDelegate(
    const ToolParams& params) const {
  if (params.Get<bool>("use_dummy_delegate")) {
    auto default_options = TfLiteDummyDelegateOptionsDefault();
    return TfLiteDummyDelegateCreateUnique(&default_options);
  }
  return TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});
}

std::pair<TfLiteDelegatePtr, int>
DummyDelegateProvider::CreateRankedTfLiteDelegate(
    const ToolParams& params) const {
  auto ptr = CreateTfLiteDelegate(params);
  return std::make_pair(std::move(ptr),
                        params.GetPosition<bool>("use_dummy_delegate"));
}
}  // namespace tools
}  // namespace tflite
