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
#ifndef MACHINA_CC_EXPERIMENTAL_LIBEXPORT_SAVE_H_
#define MACHINA_CC_EXPERIMENTAL_LIBEXPORT_SAVE_H_

#include <string>

#include "machina/core/platform/status.h"

namespace machina {
namespace libexport {

// Writes a saved model to disk.
//
// Writes a saved model to the given `export_dir`.
TF_EXPORT absl::Status Save(const std::string& export_dir);

}  // namespace libexport
}  // namespace machina

#endif  // MACHINA_CC_EXPERIMENTAL_EXPORT_EXPORT_H_
