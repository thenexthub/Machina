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

#include "machina/core/util/activation_mode.h"

#include "absl/status/status.h"
#include "machina/core/framework/node_def_util.h"
#include "machina/core/lib/core/errors.h"

namespace machina {

absl::Status GetActivationModeFromString(const string& str_value,
                                         ActivationMode* value) {
  if (str_value == "None") {
    *value = NONE;
  } else if (str_value == "Sigmoid") {
    *value = SIGMOID;
  } else if (str_value == "Relu") {
    *value = RELU;
  } else if (str_value == "Relu6") {
    *value = RELU6;
  } else if (str_value == "ReluX") {
    *value = RELUX;
  } else if (str_value == "Tanh") {
    *value = TANH;
  } else if (str_value == "BandPass") {
    *value = BANDPASS;
  } else {
    return errors::NotFound(str_value, " is not an allowed activation mode");
  }
  return absl::OkStatus();
}

}  // end namespace machina
