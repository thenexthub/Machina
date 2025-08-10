/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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

#ifndef MACHINA_CORE_PLATFORM_ENV_H_
#define MACHINA_CORE_PLATFORM_ENV_H_

#include <stdint.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "machina/core/platform/env_time.h"
#include "machina/core/platform/errors.h"
#include "machina/core/platform/file_system.h"
#include "machina/core/platform/macros.h"
#include "machina/core/platform/mutex.h"
#include "machina/core/platform/numa.h"
#include "machina/core/platform/platform.h"
#include "machina/core/platform/protobuf.h"
#include "machina/core/platform/status.h"
#include "machina/core/platform/stringpiece.h"
#include "machina/core/platform/types.h"
#include "tsl/platform/env.h"

namespace machina {
// NOLINTBEGIN(misc-unused-using-decls)
using tsl::Env;
using tsl::EnvWrapper;
using tsl::FileSystemCopyFile;
using tsl::ReadBinaryProto;
using tsl::ReadFileToString;
using tsl::ReadTextOrBinaryProto;
using tsl::ReadTextProto;
using tsl::setenv;
using tsl::Thread;
using tsl::ThreadOptions;
using tsl::unsetenv;
using tsl::WriteBinaryProto;
using tsl::WriteStringToFile;
using tsl::WriteTextProto;
namespace register_file_system {
using tsl::register_file_system::Register;
}  // namespace register_file_system
// NOLINTEND(misc-unused-using-decls)
}  // namespace machina

#endif  // MACHINA_CORE_PLATFORM_ENV_H_
