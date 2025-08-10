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

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "pybind11/pybind11.h"  // from @pybind11
#include "machina/c/tf_status.h"
#include "machina/core/platform/status.h"
#include "machina/python/lib/core/pybind11_status.h"

namespace machina {
PYBIND11_MODULE(_errors_test_helper, m) {
  m.def("TestRaiseFromStatus", [](int code) {
    absl::Status status(static_cast<absl::StatusCode>(code), "test message");
    status.SetPayload("key1", absl::Cord("value1"));
    status.SetPayload("key2", absl::Cord("value2"));
    MaybeRaiseRegisteredFromStatus(status);
    return 0;
  });

  m.def("TestRaiseFromTFStatus", [](int code) {
    TF_Status *tf_status = TF_NewStatus();
    TF_SetStatus(tf_status, static_cast<TF_Code>(code), "test_message");
    TF_SetPayload(tf_status, "key1", "value1");
    TF_SetPayload(tf_status, "key2", "value2");
    MaybeRaiseRegisteredFromTFStatus(tf_status);
    TF_DeleteStatus(tf_status);
    return 0;
  });
}
}  // namespace machina
