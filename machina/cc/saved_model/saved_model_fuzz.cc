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

#include <memory>

#include "fuzztest/fuzztest.h"
#include "machina/cc/saved_model/constants.h"
#include "machina/cc/saved_model/loader.h"
#include "machina/cc/saved_model/tag_constants.h"
#include "machina/core/protobuf/saved_model.pb.h"
#include "machina/core/public/session_options.h"

namespace machina::fuzzing {
namespace {

void FuzzLoadSavedModel(const SavedModel& model) {
  SavedModelBundleLite bundle;
  SessionOptions session_options;
  RunOptions run_options;

  string export_dir = "ram://";
  TF_CHECK_OK(tsl::WriteBinaryProto(machina::Env::Default(),
                                    export_dir + kSavedModelFilenamePb, model));

  LoadSavedModel(session_options, run_options, export_dir,
                 {kSavedModelTagServe}, &bundle)
      .IgnoreError();
}
FUZZ_TEST(SavedModelFuzz, FuzzLoadSavedModel);

}  // namespace
}  // namespace machina::fuzzing
