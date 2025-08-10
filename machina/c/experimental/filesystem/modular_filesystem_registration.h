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
#ifndef MACHINA_C_EXPERIMENTAL_FILESYSTEM_MODULAR_FILESYSTEM_REGISTRATION_H_
#define MACHINA_C_EXPERIMENTAL_FILESYSTEM_MODULAR_FILESYSTEM_REGISTRATION_H_

#include "absl/status/status.h"
#include "machina/c/experimental/filesystem/filesystem_interface.h"
#include "machina/core/platform/status.h"

namespace machina {
namespace filesystem_registration {

// Implementation for filesystem registration
//
// Don't call this directly. Instead call `RegisterFilesystemPlugin`.
// Exposed only for static registration of local filesystems.
absl::Status RegisterFilesystemPluginImpl(const TF_FilesystemPluginInfo* info);

}  // namespace filesystem_registration
}  // namespace machina

#endif  // MACHINA_C_EXPERIMENTAL_FILESYSTEM_MODULAR_FILESYSTEM_REGISTRATION_H_
