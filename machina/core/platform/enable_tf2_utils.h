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

#ifndef MACHINA_CORE_PLATFORM_ENABLE_TF2_UTILS_H_
#define MACHINA_CORE_PLATFORM_ENABLE_TF2_UTILS_H_

namespace machina {

// Sets the tf2 execution state. This can be used to indicate whether the user
// has explicitly asked for tf2 execution.
void set_tf2_execution(bool enabled);

// Returns true or false depending on whether the user flag for tf2 execution
// has been set. The default is false.
bool tf2_execution_enabled();

}  // namespace machina

#endif  // MACHINA_CORE_PLATFORM_ENABLE_TF2_UTILS_H_
