/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

#include "machina/lite/testing/string_util.h"

#include <memory>

#include "absl/strings/escaping.h"
#include "machina/lite/python/interpreter_wrapper/numpy.h"
#include "machina/lite/python/interpreter_wrapper/python_utils.h"
#include "machina/lite/string_type.h"
#include "machina/lite/string_util.h"

namespace tflite {
namespace testing {
namespace python {

PyObject* SerializeAsHexString(PyObject* value) {
  DynamicBuffer dynamic_buffer;
  if (!python_utils::FillStringBufferWithPyArray(value, &dynamic_buffer)) {
    return nullptr;
  }

  char* char_buffer = nullptr;
  size_t size = dynamic_buffer.WriteToBuffer(&char_buffer);
  string s = absl::BytesToHexString({char_buffer, size});
  free(char_buffer);

  return python_utils::ConvertToPyString(s.data(), s.size());
}

}  // namespace python
}  // namespace testing
}  // namespace tflite
