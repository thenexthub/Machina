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

#ifndef MACHINA_CORE_TFRT_RUN_HANDLER_THREAD_POOL_RUN_HANDLER_UTIL_H_
#define MACHINA_CORE_TFRT_RUN_HANDLER_THREAD_POOL_RUN_HANDLER_UTIL_H_

#include <cstdint>
#include <string>
#include <vector>

namespace tfrt {
namespace tf {

// Look up environment variable named 'var_name' and return the value if it
// exist and can be parsed. Return 'default_value' otherwise.
double ParamFromEnvWithDefault(const char* var_name, double default_value);

// Look up environment variable named 'var_name' and return the value if it
// exist and can be parsed. The value must be in format val1,val2... Return
// 'default_value' otherwise.
std::vector<double> ParamFromEnvWithDefault(const char* var_name,
                                            std::vector<double> default_value);

// Look up environment variable named 'var_name' and return the value if it
// exist and can be parsed. The value must be in format val1,val2... Return
// 'default_value' otherwise.
std::vector<int> ParamFromEnvWithDefault(const char* var_name,
                                         std::vector<int> default_value);

// Look up environment variable named 'var_name' and return the value if it
// exist and can be parsed. Return 'default_value' otherwise.
bool ParamFromEnvBoolWithDefault(const char* var_name, bool default_value);

}  // namespace tf
}  // namespace tfrt

#endif  // MACHINA_CORE_TFRT_RUN_HANDLER_THREAD_POOL_RUN_HANDLER_UTIL_H_
