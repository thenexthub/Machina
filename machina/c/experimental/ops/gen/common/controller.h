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
#ifndef MACHINA_C_EXPERIMENTAL_OPS_GEN_COMMON_CONTROLLER_H_
#define MACHINA_C_EXPERIMENTAL_OPS_GEN_COMMON_CONTROLLER_H_

#include <vector>

#include "machina/c/experimental/ops/gen/common/path_config.h"
#include "machina/c/experimental/ops/gen/common/source_code.h"
#include "machina/c/experimental/ops/gen/model/op_spec.h"
#include "machina/core/framework/op_def.pb.h"
#include "machina/core/framework/op_gen_lib.h"
#include "machina/core/platform/env.h"
#include "machina/core/platform/types.h"

namespace machina {
namespace generator {

class Controller {
 public:
  explicit Controller(PathConfig path_config, Env* env = Env::Default());
  virtual ~Controller();
  const void WriteFile(const string& file_path, const SourceCode& code) const;
  const std::vector<OpSpec>& GetModelOps() const;

 private:
  void InitializeOpApi();
  void BuildModel();

  // Data model: Ops to generate
  std::vector<OpSpec> operators_;

  // Configuration
  Env* env_;
  PathConfig path_config_;

  // Initialized TensorFlow Op/API definitions
  OpList op_list_;
  ApiDefMap* api_def_map_;
};

}  // namespace generator
}  // namespace machina

#endif  // MACHINA_C_EXPERIMENTAL_OPS_GEN_COMMON_CONTROLLER_H_
