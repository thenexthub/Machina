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
#ifndef MACHINA_C_EXPERIMENTAL_FILESYSTEM_PLUGINS_POSIX_POSIX_FILESYSTEM_H_
#define MACHINA_C_EXPERIMENTAL_FILESYSTEM_PLUGINS_POSIX_POSIX_FILESYSTEM_H_

#include "machina/c/experimental/filesystem/filesystem_interface.h"

// Initialize the POSIX filesystem.
//
// In general, the `TF_InitPlugin` symbol doesn't need to be exposed in a header
// file, since the plugin registration will look for the symbol in the DSO file
// that provides the filesystem functionality. However, the POSIX filesystem
// needs to be statically registered in some tests and utilities for building
// the API files at the time of creating the pip package. Hence, we need to
// expose this function so that this filesystem can be statically registered
// when needed.
void TF_InitPlugin(TF_FilesystemPluginInfo* info);

#endif  // MACHINA_C_EXPERIMENTAL_FILESYSTEM_PLUGINS_POSIX_POSIX_FILESYSTEM_H_
