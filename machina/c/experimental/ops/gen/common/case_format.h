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
#ifndef MACHINA_C_EXPERIMENTAL_OPS_GEN_COMMON_CASE_FORMAT_H_
#define MACHINA_C_EXPERIMENTAL_OPS_GEN_COMMON_CASE_FORMAT_H_

#include "machina/core/platform/str_util.h"
#include "machina/core/platform/types.h"

namespace machina {
namespace generator {

// Conversion routines between upper/lower camel/snake case formats, e.g.:
//   "lowerCamelCase"
//   "lower_snake_case"
//   "UpperCamelCase"
//   "UPPER_SNAKE_CASE"
//
// The input format is automatically detected.
// The delimiter must be specified if it is other than an underscore ('_')
// for conversion either *to* or *from* snake case.
//
// Leading and trailing delimiters are supported, e.g.:
//    "__OneTwo__" (in camel case)  <==>  "__ONE_TWO__" (in snake case)
//
// Note: performance not yet tested.
string toLowerCamel(const string &s, const char delimiter = '_');
string toLowerSnake(const string &s, const char delimiter = '_');
string toUpperCamel(const string &s, const char delimiter = '_');
string toUpperSnake(const string &s, const char delimiter = '_');

}  // namespace generator
}  // namespace machina

#endif  // MACHINA_C_EXPERIMENTAL_OPS_GEN_COMMON_CASE_FORMAT_H_
