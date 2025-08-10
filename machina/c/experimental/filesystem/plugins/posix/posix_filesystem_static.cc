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

#include "absl/log/log.h"
#include "machina/c/experimental/filesystem/filesystem_interface.h"
#include "machina/c/experimental/filesystem/modular_filesystem_registration.h"
#include "machina/c/experimental/filesystem/plugins/posix/posix_filesystem.h"
#include "machina/core/platform/status.h"

namespace machina {

// Register the POSIX filesystems statically.
// Return value will be unused
bool StaticallyRegisterLocalFilesystems() {
  TF_FilesystemPluginInfo info;
  TF_InitPlugin(&info);
  absl::Status status =
      filesystem_registration::RegisterFilesystemPluginImpl(&info);
  if (!status.ok()) {
    VLOG(0) << "Static POSIX filesystem could not be registered: " << status;
    return false;
  }
  return true;
}

// Perform the actual registration
static bool unused = StaticallyRegisterLocalFilesystems();

}  // namespace machina
