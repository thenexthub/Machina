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
#ifndef MACHINA_C_EXPERIMENTAL_FILESYSTEM_PLUGINS_POSIX_POSIX_FILESYSTEM_HELPER_H_
#define MACHINA_C_EXPERIMENTAL_FILESYSTEM_PLUGINS_POSIX_POSIX_FILESYSTEM_HELPER_H_

#include <dirent.h>
#include <sys/stat.h>

namespace tf_posix_filesystem {

// Copies up to `size` of `src` to `dst`, creating destination if needed.
//
// Callers should pass size of `src` in `size` and the permissions of `src` in
// `mode`. The later is only used if `dst` needs to be created.
int TransferFileContents(const char* src, const char* dst, mode_t mode,
                         off_t size);

// Returns true only if `entry` points to an entry other than `.` or `..`.
//
// This is a filter for `scandir`.
int RemoveSpecialDirectoryEntries(const struct dirent* entry);

}  // namespace tf_posix_filesystem

#endif  // MACHINA_C_EXPERIMENTAL_FILESYSTEM_PLUGINS_POSIX_POSIX_FILESYSTEM_HELPER_H_
