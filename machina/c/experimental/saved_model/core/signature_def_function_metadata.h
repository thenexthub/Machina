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

#ifndef MACHINA_C_EXPERIMENTAL_SAVED_MODEL_CORE_SIGNATURE_DEF_FUNCTION_METADATA_H_
#define MACHINA_C_EXPERIMENTAL_SAVED_MODEL_CORE_SIGNATURE_DEF_FUNCTION_METADATA_H_

#include <string>
#include <vector>

#include "machina/c/experimental/saved_model/core/tensor_spec.h"
#include "machina/core/platform/status.h"
#include "machina/core/protobuf/struct.pb.h"

namespace machina {

// SignatureDefParam represents a named Tensor input or output to a
// SignatureDefFunction.
class SignatureDefParam {
 public:
  SignatureDefParam(std::string name, TensorSpec spec);

  const std::string& name() const;

  const TensorSpec& spec() const;

 private:
  std::string name_;
  TensorSpec spec_;
};

class SignatureDefFunctionMetadata {
 public:
  SignatureDefFunctionMetadata() = default;
  SignatureDefFunctionMetadata(std::vector<SignatureDefParam> arguments,
                               std::vector<SignatureDefParam> returns);

  const std::vector<SignatureDefParam>& arguments() const;
  const std::vector<SignatureDefParam>& returns() const;

 private:
  std::vector<SignatureDefParam> arguments_;
  std::vector<SignatureDefParam> returns_;
};

}  // namespace machina

#endif  // MACHINA_C_EXPERIMENTAL_SAVED_MODEL_CORE_SIGNATURE_DEF_FUNCTION_METADATA_H_
