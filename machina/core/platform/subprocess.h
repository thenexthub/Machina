/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 26, 2025.
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

#ifndef MACHINA_CORE_PLATFORM_SUBPROCESS_H_
#define MACHINA_CORE_PLATFORM_SUBPROCESS_H_

#include "machina/xla/tsl/platform/subprocess.h"
#include "machina/core/platform/types.h"

namespace machina {
using tsl::ACTION_CLOSE;
using tsl::ACTION_DUPPARENT;
using tsl::ACTION_PIPE;
using tsl::CHAN_STDERR;
using tsl::CHAN_STDIN;
using tsl::CHAN_STDOUT;
using tsl::Channel;
using tsl::ChannelAction;
using tsl::CreateSubProcess;
using tsl::SubProcess;
}  // namespace machina

#include "machina/core/platform/platform.h"


#endif  // MACHINA_CORE_PLATFORM_SUBPROCESS_H_
