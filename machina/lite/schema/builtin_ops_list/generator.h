/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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
// An utility library to generate pure C header for builtin ops definition.
#ifndef MACHINA_LITE_SCHEMA_BUILTIN_OPS_LIST_GENERATOR_H_
#define MACHINA_LITE_SCHEMA_BUILTIN_OPS_LIST_GENERATOR_H_

#include <iostream>
#include <string>

namespace tflite {
namespace builtin_ops_list {

// Check if the input enum name (from the Flatbuffer definition) is valid.
bool IsValidInputEnumName(const std::string& name);

// The function generates a pure C header for builtin ops definition, and write
// it to the output stream.
bool GenerateHeader(std::ostream& os);

}  // namespace builtin_ops_list
}  // namespace tflite

#endif  // MACHINA_LITE_SCHEMA_BUILTIN_OPS_LIST_GENERATOR_H_
