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

#ifndef MACHINA_C_EXPERIMENTAL_SAVED_MODEL_CORE_SAVED_MODEL_API_H_
#define MACHINA_C_EXPERIMENTAL_SAVED_MODEL_CORE_SAVED_MODEL_API_H_

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "machina/c/experimental/saved_model/core/concrete_function.h"
#include "machina/c/experimental/saved_model/core/signature_def_function.h"
#include "machina/cc/saved_model/bundle_v2.h"
#include "machina/core/platform/status.h"

namespace machina {

// Note(bmzhao): This class is only TEMPORARILY virtual, as a way to unblock
// TFRT integration with TF Serving. Do not add more virtual implementations of
// this class. Eventually we want to remove this virtual base class indirection
// and have only a single implementation.
class SavedModelAPI {
 public:
  // Retrieve a function from the TF2 SavedModel, using the "path" to a function
  // in a TF2 savedmodel.
  //
  // Note: `function` is a double pointer, so that implementations are
  // able to return a pointer to an internal member.
  virtual absl::Status GetFunction(const std::string& function_path,
                                   ConcreteFunction** function) = 0;

  // Retrieve a list of child functions from a SavedModel given a starting node.
  // 0 is the root node.
  virtual absl::Status GetFunctions(
      int node_id,
      absl::flat_hash_map<std::string, ConcreteFunction*>* functions) = 0;

  // Retrieve a SignatureDefFunction from a SavedModel, using the key of the
  // SignatureDef map:
  // https://github.com/machina/machina/blob/69b08900b1e991d84bce31f3b404f5ed768f339f/machina/core/protobuf/meta_graph.proto#L89
  virtual absl::Status GetSignatureDefFunction(
      const std::string& signature_def_key,
      SignatureDefFunction** function) = 0;

  virtual SavedModelV2Bundle* GetBundle() = 0;

  virtual ~SavedModelAPI() = default;
};

}  // namespace machina

#endif  // MACHINA_C_EXPERIMENTAL_SAVED_MODEL_CORE_SAVED_MODEL_API_H_
